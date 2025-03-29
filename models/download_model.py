from huggingface_hub import snapshot_download


# Download student model
# Model repo on Hugging Face
model_name = "meta-llama/Llama-3.2-3B"
local_dir="models/Llama-3.2-3B"

# Download the entire model into the current directory
snapshot_download(repo_id=model_name, local_dir=local_dir)
print(f"Model {model_name} downloaded to {local_dir}")

# Download teacher model
# Model repo on Hugging Face
model_name = "meta-llama/Llama-2-7b-hf"
local_dir="models/Llama-2-7b-hf"

# Download the entire model into the current directory
snapshot_download(repo_id=model_name, local_dir=local_dir)

print(f"Model {model_name} downloaded to {local_dir}")