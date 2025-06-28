import os
import time
import torch
from PIL import Image
from cog import BasePredictor, Input  # Path removed
import requests # Added
import tempfile # Added

from flux.sampling import denoise, get_schedule, prepare_kontext, unpack
from flux.util import (
    configs,
    load_clip,
    load_t5
)
from flux.model import Flux
from flux.modules.autoencoder import AutoEncoder
from safetensors.torch import load_file as load_sft
from safety_checker import SafetyChecker
from util import print_timing, generate_compute_step_map
from weights import download_weights

from flux.util import ASPECT_RATIOS

# Kontext model configuration
KONTEXT_WEIGHTS_URL = "https://weights.replicate.delivery/default/black-forest-labs/kontext/release-candidate/kontext-dev.sft"
KONTEXT_WEIGHTS_PATH = "./models/kontext/kontext-dev.sft"
AE_WEIGHTS_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/safetensors/ae.safetensors"
AE_WEIGHTS_PATH = "./models/flux-dev/ae.safetensors"
T5_WEIGHTS_URL = "https://weights.replicate.delivery/default/official-models/flux/t5/t5-v1_1-xxl.tar"
T5_WEIGHTS_PATH = "./models/t5"
CLIP_URL = "https://weights.replicate.delivery/default/official-models/flux/clip/clip-vit-large-patch14.tar"
CLIP_PATH = "./models/clip"

TORCH_COMPILE_CACHE = "./torch-compile-cache-flux-dev-kontext.bin"

class FluxDevKontextPredictor(BasePredictor):
    """
    Flux.1 Kontext Predictor - Image-to-image transformation model using FLUX.1-dev architecture
    """

    def setup(self) -> None:
        """Load model weights and initialize the pipeline"""
        self.device = torch.device("cuda")

        # Download all weights if needed
        download_model_weights()

        # Initialize models
        st = time.time()
        print("Loading t5...")
        self.t5 = load_t5(self.device, max_length=512, t5_path=T5_WEIGHTS_PATH)
        print(f"Loaded t5 in {time.time() - st} seconds")
        st = time.time()
        self.clip = load_clip(self.device, clip_path=CLIP_PATH)
        print(f"Loaded clip in {time.time() - st} seconds")
        st = time.time()
        self.model = load_kontext_model(device=self.device)
        print(f"Loaded kontext model in {time.time() - st} seconds")
        st = time.time()
        self.ae = load_ae_local(device=self.device)
        print(f"Loaded ae in {time.time() - st} seconds")
        st = time.time()
        self.model = torch.compile(self.model, dynamic=True)

        # Initialize safety checker

        # Compile the model
        print("Compiling model with torch.compile...")
        start_time = time.time()
        self.model = torch.compile(self.model, dynamic=True)
        self._warmup_model() # Call after compilation

        # Initialize safety checker
        self.safety_checker = SafetyChecker()
        print(f"Compilation and warmup finished in {time.time() - start_time} seconds")
        print("FluxDevKontextPredictor setup complete")

    def _warmup_model(self):
        """Dedicated warmup method to compile self.model by calling it directly."""
        print("Starting model compilation with dummy inputs...")
        try:
            config_spec = configs["flux-dev"]
            params = config_spec.params

            bs = 1
            # For 1MP image (1024x1024), effective latent dims H_eff=64, W_eff=64 (after AE stride 16)
            # Sequence length for packed image latents: (64/2)*(64/2) = 1024
            seq_len_img = (64 // 2) * (64 // 2)

            dummy_img_latent_patches = torch.randn(bs, seq_len_img, params.in_channels, device=self.device, dtype=torch.bfloat16)
            dummy_img_ids_positions = torch.randn(bs, seq_len_img, 3, device=self.device, dtype=torch.bfloat16)

            seq_len_txt = self.t5.max_length # Typically 512 for T5-XXL
            dummy_txt_embeddings = torch.randn(bs, seq_len_txt, params.context_in_dim, device=self.device, dtype=torch.bfloat16)
            dummy_txt_ids_positions = torch.randn(bs, seq_len_txt, 3, device=self.device, dtype=torch.bfloat16)

            dummy_clip_vector = torch.randn(bs, params.vec_in_dim, device=self.device, dtype=torch.bfloat16)
            dummy_timesteps = torch.tensor([0.5], device=self.device, dtype=torch.bfloat16)
            dummy_guidance_strength = torch.tensor([4.0], device=self.device, dtype=torch.bfloat16)

            # For Kontext, conditioning image latents are concatenated
            # Assume conditioning image has same latent dimensions for warmup
            seq_len_img_cond = seq_len_img
            dummy_img_cond_latent_patches = torch.randn(bs, seq_len_img_cond, params.in_channels, device=self.device, dtype=torch.bfloat16)
            dummy_img_cond_ids_positions = torch.randn(bs, seq_len_img_cond, 3, device=self.device, dtype=torch.bfloat16)

            final_dummy_img_input = torch.cat((dummy_img_latent_patches, dummy_img_cond_latent_patches), dim=1)
            final_dummy_img_ids_input = torch.cat((dummy_img_ids_positions, dummy_img_cond_ids_positions), dim=1)

            # Call the compiled model
            _ = self.model(
                img=final_dummy_img_input,
                img_ids=final_dummy_img_ids_input,
                txt=dummy_txt_embeddings,
                txt_ids=dummy_txt_ids_positions,
                y=dummy_clip_vector,
                timesteps=dummy_timesteps,
                guidance=dummy_guidance_strength,
            )
            print("Model compilation with dummy inputs successful.")
        except Exception as e:
            print(f"Error during model compilation with dummy inputs: {e}")
            raise


    def predict(
        self,
        prompt: str = Input(
            description="Text description of what you want to generate, or the instruction on how to edit the given image.",
        ),
        input_image: str = Input(
            description="Image URL to use as reference. Must be jpeg, png, gif, or webp.",
        ),
        aspect_ratio: str = Input
            description="Aspect ratio of the generated image. Use 'match_input_image' to match the aspect ratio of the input image.",
            choices=list(ASPECT_RATIOS.keys()),
            default="match_input_image",
        ),
        # megapixels: str = Input(
        #     description="Approximate number of megapixels for generated image",
        #     choices=["1", "0.25"],
        #     default="1",
        # ),
        num_inference_steps: int = Input(
            description="Number of inference steps", default=28, ge=4, le=50
        ),
        guidance: float = Input(
            description="Guidance scale for generation", default=2.5, ge=0.0, le=10.0
        ),
        seed: int = Input(
            description="Random seed for reproducible generation. Leave blank for random.",
            default=None,
        ),
        output_format: str = Input(
            description="Output image format",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
            default=80,
            ge=0,
            le=100,
        ),
        disable_safety_checker: bool = Input(
            description="Disable NSFW safety checker", default=False
        ),
        go_fast: bool = Input(
            description="Make the model go fast, output quality may be slightly degraded for more difficult prompts",
            default=True,
        ),
    ) -> Path:
        """
        Generate an image based on the text prompt and conditioning image using FLUX.1 Kontext
        """
        with torch.inference_mode(), print_timing("generate image"):
            seed = prepare_seed(seed)

            if aspect_ratio == "match_input_image":
                target_width, target_height = None, None
            else:
                target_width, target_height = ASPECT_RATIOS[aspect_ratio]

            # Download image from URL and save to a temporary file
            try:
                response = requests.get(input_image, stream=True)
                response.raise_for_status()  # Raise an exception for bad status codes

                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        tmp_file.write(chunk)
                    tmp_image_path = tmp_file.name
            except requests.exceptions.RequestException as e:
                raise Exception(f"Failed to download image from URL: {input_image}. Error: {e}")

            try:
                # Prepare input for kontext sampling
                inp, final_height, final_width = prepare_kontext(
                    t5=self.t5,
                    clip=self.clip,
                    prompt=prompt,
                    ae=self.ae,
                    img_cond_path=tmp_image_path, # Use downloaded image path
                    target_width=target_width,
                    target_height=target_height,
                    bs=1,
                    seed=seed,
                    device=self.device,
                )
            finally:
                # Clean up the temporary image file
                if os.path.exists(tmp_image_path):
                    os.remove(tmp_image_path)
            
            if go_fast:
                compute_step_map = generate_compute_step_map("go really fast", num_inference_steps)
            else:
                compute_step_map = generate_compute_step_map("none", num_inference_steps)

            # Remove the original conditioning image from memory to save space
            inp.pop("img_cond_orig", None)

            # Get sampling schedule
            timesteps = get_schedule(
                num_inference_steps,
                inp["img"].shape[1],
                shift=True,  # flux-dev uses shift=True
            )

            # Generate image
            x = denoise(self.model, **inp, timesteps=timesteps, guidance=guidance, compute_step_map=compute_step_map)

            # Decode latents to pixel space
            x = unpack(x.float(), final_height, final_width)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                x = self.ae.decode(x)

            # Convert to image
            x = x.clamp(-1, 1)
            x = (x + 1) / 2
            x = (x.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()
            image = Image.fromarray(x[0])

            # Apply safety checking
            if not disable_safety_checker:
                images = self.safety_checker.filter_images([image])
                if not images:
                    raise Exception(
                        "Generated image contained NSFW content. Try running it again with a different prompt."
                    )
                image = images[0]

            # Save image
            output_path = f"output.{output_format}"
            if output_format == "png":
                image.save(output_path)
            elif output_format == "webp":
                image.save(
                    output_path, format="WEBP", quality=output_quality, optimize=True
                )
            else:  # jpg
                image.save(
                    output_path, format="JPEG", quality=output_quality, optimize=True
                )

            # Return the output path
            return output_path # Return string path instead of Path object


def download_model_weights():
    """Download all required model weights if they don't exist"""
    from pathlib import Path # Import Path here for download_weights
    # Download kontext weights
    if not os.path.exists(KONTEXT_WEIGHTS_PATH):
        print("Kontext weights not found, downloading...")
        download_weights(KONTEXT_WEIGHTS_URL, Path(KONTEXT_WEIGHTS_PATH))
        print("Kontext weights downloaded successfully")
    else:
        print("Kontext weights already exist")

    # Download autoencoder weights
    if not os.path.exists(AE_WEIGHTS_PATH):
        print("Autoencoder weights not found, downloading...")
        download_weights(AE_WEIGHTS_URL, Path(AE_WEIGHTS_PATH))
        print("Autoencoder weights downloaded successfully")
    else:
        print("Autoencoder weights already exist")

    if not os.path.exists(T5_WEIGHTS_PATH):
        print("T5 weights not found, downloading...")
        download_weights(T5_WEIGHTS_URL, Path(T5_WEIGHTS_PATH))
        print("T5 weights downloaded successfully")
    else:
        print("T5 weights already exist")
        
    if not os.path.exists(CLIP_PATH):
        print("CLIP weights not found, downloading...")
        download_weights(CLIP_URL, Path(CLIP_PATH))
        print("CLIP weights downloaded successfully")
    else:
        print("CLIP weights already exist")


def load_kontext_model(device: str | torch.device = "cuda"):
    """Load the kontext model with complete transformer weights"""
    # Use flux-dev config as base for kontext model
    config = configs["flux-dev"]

    print("Loading kontext model...")
    with torch.device("meta"):
        model = Flux(config.params).to(torch.bfloat16)

    # Load kontext weights (complete transformer)
    print(f"Loading kontext weights from {KONTEXT_WEIGHTS_PATH}")
    sd = load_sft(KONTEXT_WEIGHTS_PATH, device=str(device))
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)

    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")

    return model


def load_ae_local(device: str | torch.device = "cuda"):
    """Load autoencoder from local weights"""
    config = configs["flux-dev"]

    print("Loading autoencoder...")
    with torch.device("meta"):
        ae = AutoEncoder(config.ae_params)

    print(f"Loading autoencoder weights from {AE_WEIGHTS_PATH}")
    sd = load_sft(AE_WEIGHTS_PATH, device=str(device))
    missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)

    if missing:
        print(f"AE Missing keys: {missing}")
    if unexpected:
        print(f"AE Unexpected keys: {unexpected}")

    return ae


def prepare_seed(seed: int) -> int:
    if not seed:
        seed = int.from_bytes(os.urandom(2), "big")
    print(f"Using seed: {seed}")
    return seed
