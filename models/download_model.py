from huggingface_hub import hf_hub_download


# Model repo on Hugging Face
model_name = "QuantFactory/TinyLlama_v1.1-GGUF"
file = "TinyLlama_v1.1.Q2_K.gguf"
file_path = hf_hub_download(repo_id=model_name, filename=file, local_dir="TinyLlama_v1.1-GGUF")
print(f"Downloaded {file} to: {file_path}")

# Download config.json from original tinyllama
repo_id = "TinyLlama/TinyLlama_v1.1"
file = "config.json"
file_path = hf_hub_download(repo_id=repo_id, filename=file, local_dir="TinyLlama_v1.1-GGUF")
print(f"Downloaded {file} to: {file_path}")