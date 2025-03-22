from huggingface_hub import snapshot_download

# Model repo on Hugging Face
model_name = "meta-llama/Llama-3.2-3B"

# Download the entire model into the current directory
snapshot_download(repo_id=model_name, local_dir="models/Llama-3.2-3B")

print(f"Model {model_name} downloaded to models/Llama-3.2-3B")