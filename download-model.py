import os
from faster_whisper import download_model

model_cache_dir = os.getenv("MODEL_CACHE_DIR", "./models")
model_type = os.getenv('WHISPER_MODEL', 'large-v3')

def preload_models():
    model_dir_name = model_type.replace("/", "_")
    output_dir = os.path.join(model_cache_dir, model_dir_name)

    print(f"Downloading Whisper model to {output_dir}...")
    
    model_path = download_model(model_type, output_dir=output_dir)
    
    print(f"Model downloaded successfully to: {model_path}")

if __name__ == "__main__":
    preload_models()