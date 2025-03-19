from huggingface_hub import snapshot_download

# Model repo on Hugging Face
model_name = "meta-llama/Llama-2-7b-hf"

# Download the entire model into the current directory
snapshot_download(repo_id=model_name, local_dir="models/Llama-2-7b-hf")

print(f"Model {model_name} downloaded to models/Llama-2-7b-hf")