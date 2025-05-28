from contextlib import contextmanager
import time
import os
import sys
import subprocess
import torch
from PIL import Image
from cog import BasePredictor, Path, Input
from transformers import pipeline

from flux.sampling import denoise, get_schedule, prepare_kontext, unpack
from flux.util import (
    configs,
    load_ae,
    load_clip,
    load_t5,
    save_image,
    PREFERED_KONTEXT_RESOLUTIONS,
)
from flux.model import Flux
from safetensors.torch import load_file as load_sft

# Environment setup
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torch-inductor-cache-kontext"

# Kontext model configuration
KONTEXT_WEIGHTS_URL = "https://weights.replicate.delivery/default/black-forest-labs/kontext/pre-release/preliminary-dev-kontext.sft"
KONTEXT_WEIGHTS_PATH = "/models/kontext/preliminary-dev-kontext.sft"

# Model weights URLs
AE_WEIGHTS_URL = (
    "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/safetensors/ae.safetensors"
)
AE_WEIGHTS_PATH = "/models/flux-dev/ae.safetensors"

MAX_IMAGE_SIZE = 1440

ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "21:9": (1536, 640),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "4:5": (896, 1088),
    "5:4": (1088, 896),
    "3:4": (896, 1152),
    "4:3": (1152, 896),
    "9:16": (768, 1344),
    "9:21": (640, 1536),
}


class FluxDevKontextPredictor(BasePredictor):
    """
    Flux.1 Kontext Predictor - Image-to-image transformation model using FLUX.1-dev architecture
    """

    def setup(self) -> None:
        """Load model weights and initialize the pipeline"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Download all weights if needed
        download_weights()

        # Initialize models
        self.t5 = load_t5(self.device, max_length=512)
        self.clip = load_clip(self.device)
        self.model = load_kontext_model(device=self.device)
        self.ae = load_ae_local(device=self.device)

        # Compile models for faster execution
        print("Compiling models with torch.compile...")
        self.model = torch.compile(self.model, mode="max-autotune")
        self.ae.decode = torch.compile(self.ae.decode, mode="max-autotune")

        # Initialize NSFW classifier
        self.nsfw_classifier = pipeline(
            "image-classification", model="Falconsai/nsfw_image_detection", device=self.device
        )

        print("FluxDevKontextPredictor setup complete")

    def size_from_aspect_megapixels(self, aspect_ratio: str, megapixels: str = "1") -> tuple[int, int]:
        """Convert aspect ratio and megapixels to width and height"""
        width, height = ASPECT_RATIOS[aspect_ratio]
        if megapixels == "0.25":
            width, height = width // 2, height // 2
        return (width, height)

    def predict(
        self,
        prompt: str = Input(
            description="Text prompt describing the desired transformation",
            default="replace the logo with the text 'Hello World'",
        ),
        conditioning_image: Path = Input(description="Input image to condition the generation"),
        aspect_ratio: str = Input(
            description="Aspect ratio for the generated image",
            choices=list(ASPECT_RATIOS.keys()),
            default="1:1",
        ),
        megapixels: str = Input(
            description="Approximate number of megapixels for generated image",
            choices=["1", "0.25"],
            default="1",
        ),
        num_inference_steps: int = Input(description="Number of inference steps", default=30, ge=4, le=50),
        guidance: float = Input(description="Guidance scale for generation", default=2.5, ge=0.0, le=10.0),
        seed: int = Input(
            description="Random seed for reproducible generation. Use 0 for random seed.", default=0
        ),
        output_format: str = Input(
            description="Output image format", choices=["webp", "jpg", "png"], default="webp"
        ),
        disable_safety_checker: bool = Input(description="Disable NSFW safety checker", default=False),
        nsfw_threshold: float = Input(description="NSFW detection threshold", default=0.85, ge=0.0, le=1.0),
    ) -> Path:
        """
        Generate an image based on the text prompt and conditioning image using FLUX.1 Kontext
        """
        with (torch.inference_mode(), print_timing("generate image")):
            # Prepare seed
            if seed == 0:
                seed = int.from_bytes(os.urandom(2), "big")
            print(f"Using seed: {seed}")

            # Prepare target dimensions
            target_width = width if width > 0 else None
            target_height = height if height > 0 else None

            # Prepare input for kontext sampling
            print("Preparing input...")
            inp, final_height, final_width = prepare_kontext(
                t5=self.t5,
                clip=self.clip,
                prompt=prompt,
                ae=self.ae,
                img_cond_path=str(conditioning_image),
                target_width=target_width,
                target_height=target_height,
                bs=1,
                seed=seed,
                device=self.device,
            )

            # Remove the original conditioning image from memory to save space
            inp.pop("img_cond_orig", None)

            # Get sampling schedule
            timesteps = get_schedule(
                num_inference_steps,
                inp["img"].shape[1],
                shift=True,  # flux-dev uses shift=True
            )

            # Generate image
            print("Generating image...")
            x = denoise(self.model, **inp, timesteps=timesteps, guidance=guidance)

            # Decode latents to pixel space
            print("Decoding image...")
            x = unpack(x.float(), final_height, final_width)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                x = self.ae.decode(x)

            # Save image
            output_path = f"output.{output_format}"
            idx = save_image(
                nsfw_classifier=self.nsfw_classifier,
                name="flux-kontext",
                output_name=output_path,
                idx=0,
                x=x,
                add_sampling_metadata=True,
                prompt=prompt,
                nsfw_threshold=nsfw_threshold if not disable_safety_checker else 0.0,
            )

            # Return the output path
            return Path(output_path)


def download_weights():
    """Download all required weights if they don't exist"""
    # Download kontext weights
    if not os.path.exists(KONTEXT_WEIGHTS_PATH):
        print(f"Downloading kontext weights to {KONTEXT_WEIGHTS_PATH}")
        os.makedirs(os.path.dirname(KONTEXT_WEIGHTS_PATH), exist_ok=True)
        subprocess.check_call(["pget", "-f", KONTEXT_WEIGHTS_URL, KONTEXT_WEIGHTS_PATH])
        print("Kontext weights downloaded successfully")
    else:
        print("Kontext weights already exist")

    # Download autoencoder weights
    if not os.path.exists(AE_WEIGHTS_PATH):
        print(f"Downloading autoencoder weights to {AE_WEIGHTS_PATH}")
        os.makedirs(os.path.dirname(AE_WEIGHTS_PATH), exist_ok=True)
        subprocess.check_call(["pget", "-f", AE_WEIGHTS_URL, AE_WEIGHTS_PATH])
        print("Autoencoder weights downloaded successfully")
    else:
        print("Autoencoder weights already exist")


def load_kontext_model(device: str | torch.device = "cuda"):
    """Load the kontext model with complete transformer weights"""
    from flux.model import Flux

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
    from flux.modules.autoencoder import AutoEncoder

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


def make_multiple_of_16(n: int) -> int:
    """Round number to nearest multiple of 16"""
    return ((n + 15) // 16) * 16


@contextmanager
def print_timing(operation_name: str):
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        print(f"{operation_name} took {elapsed_time:.2f} seconds")
