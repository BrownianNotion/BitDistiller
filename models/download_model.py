from huggingface_hub import hf_hub_download


# Model repo on Hugging Face
model_name = "QuantFactory/TinyLlama_v1.1-GGUF"
files = ["TinyLlama_v1.1.Q2_K.gguf", "config.json"]

# Download each file
for file in files:
    file_path = hf_hub_download(repo_id=model_name, filename=file, local_dir="TinyLlama_v1.1-GGUF")
    print(f"Downloaded {file} to: {file_path}")