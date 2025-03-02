from huggingface_hub import snapshot_download

# Model repo on Hugging Face
model_name = "TinyLlama/TinyLlama_v1.1"

# Download the entire model into the current directory
snapshot_download(repo_id=model_name, local_dir="models/TinyLlama_v1.1")

print(f"Model {model_name} downloaded to models/TinyLlama_v1.1")