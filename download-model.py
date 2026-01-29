import os
from faster_whisper import download_model

model_cache_dir = os.getenv("MODEL_CACHE_DIR", "./models")
model_type = os.getenv('WHISPER_MODEL', 'large-v3')

def preload_models():
    print(f"Downloading Whisper model to {model_cache_dir}...")
    
    model_path = download_model(model_type, output_dir=model_cache_dir)
    
    print(f"Model downloaded successfully to: {model_path}")

if __name__ == "__main__":
    preload_models()