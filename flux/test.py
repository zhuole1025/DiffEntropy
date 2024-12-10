import os
import huggingface_hub

def download_model():
    # Download the model file from HuggingFace Hub
    model_path = huggingface_hub.hf_hub_download(
        repo_id="black-forest-labs/FLUX.1-Redux-dev",
        filename="flux1-redux-dev.safetensors"
    )
    return model_path

if __name__ == "__main__":
    model_path = download_model()
    print(f"Model downloaded to: {model_path}")
