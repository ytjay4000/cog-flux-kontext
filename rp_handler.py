import runpod
from predict import FluxDevKontextPredictor, download_model_weights
import os
import torch
import traceback # Added for detailed error logging

# Global predictor variable
predictor = None

try:
    # Initialize predictor globally to reuse loaded models across calls
    print("Global initialization started.")

    # Ensure models are downloaded before Predictor initialization if setup expects them
    print("Downloading models for global predictor...")
    download_model_weights() # Explicitly download weights first
    print("Model download process completed.")

    print("Initializing FluxDevKontextPredictor...")
    predictor_instance = FluxDevKontextPredictor()
    print("FluxDevKontextPredictor instance created. Calling setup...")
    predictor_instance.setup() # Call setup once during initialization
    predictor = predictor_instance # Assign to global variable only after successful setup
    print("FluxDevKontextPredictor initialized and setup complete.")

except Exception as e:
    print("!!! ERROR DURING GLOBAL INITIALIZATION !!!")
    print(f"Exception type: {type(e)}")
    print(f"Exception message: {e}")
    print("Traceback:")
    traceback.print_exc() # Prints full traceback to stdout/stderr
    # predictor remains None, handler will fail if called, or we can raise to stop worker
    raise # Re-raise to ensure worker fails clearly if init fails

def handler(event):
    """
    RunPod Serverless Handler for FLUX.1 Kontext.
    """
    print("Received event:", event)

    if predictor is None:
        # This case should ideally not be reached if __main__ block doesn't catch
        # the re-raised exception from global scope. But as a safeguard:
        print("ERROR: Predictor not initialized. Global initialization likely failed.")
        return {"error": "Predictor not initialized. Check worker startup logs."}

    job_input = event.get('input', {})

    # Extract parameters, providing defaults similar to Cog or sensible values
    prompt = job_input.get('prompt', "A beautiful landscape.")
    input_image_url = job_input.get('input_image') # This is required

    if not input_image_url:
        return {"error": "input_image URL is required."}

    aspect_ratio = job_input.get('aspect_ratio', "match_input_image")
    num_inference_steps = job_input.get('num_inference_steps', 28)
    guidance = job_input.get('guidance', 2.5)
    seed = job_input.get('seed', None) # None will lead to random seed
    output_format = job_input.get('output_format', "webp")
    output_quality = job_input.get('output_quality', 80)
    disable_safety_checker = job_input.get('disable_safety_checker', False)
    go_fast = job_input.get('go_fast', True)

    print(f"Processing job with prompt: '{prompt}' and image_url: '{input_image_url}'")

    try:
        # Call the predictor's predict method
        output_image_path = predictor.predict(
            prompt=prompt,
            input_image=input_image_url,
            aspect_ratio=aspect_ratio,
            num_inference_steps=num_inference_steps,
            guidance=guidance,
            seed=seed,
            output_format=output_format,
            output_quality=output_quality,
            disable_safety_checker=disable_safety_checker,
            go_fast=go_fast
        )

        # For RunPod, we typically return a JSON response.
        # If the output is a file path, you might want to upload it to a bucket
        # and return the URL, or return the image data directly (e.g., base64 encoded).
        # For now, returning the path as per Cog behavior, but this might need adjustment
        # based on how RunPod expects file outputs to be handled.
        # A common pattern is to return a URL to the generated image.
        # Let's assume for now the output_image_path is what's expected.
        # If direct image data is needed, we would read the file here.
        print(f"Prediction successful. Output image at: {output_image_path}")
        return {"output_image_path": output_image_path}

    except Exception as e:
        print(f"Error during prediction: {e}")
        # It's good to clear CUDA cache in case of OOM or other GPU errors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"error": str(e)}

# Start the RunPod serverless worker
if __name__ == "__main__":
    print("Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})
