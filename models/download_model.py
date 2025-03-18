from huggingface_hub import snapshot_download

# Model repo on Hugging Face
model_name = "BrownianNotion/TinyLlama_v1.1_mix_wikitext_alpaca_2bit_BitDistiller_baseline"

# Download the entire model into the current directory
#snapshot_download(repo_id=model_name, local_dir="models/TinyLlama_v1.1_mix_wikitext_alpaca_2bit_BitDistiller_baseline")

print(f"Model {model_name} downloaded to TinyLlama_v1.1_mix_wikitext_alpaca_2bit_BitDistiller_baseline")


# Download 1 bit model as well
model_name = "fredericowieser/TinyLlama_v1.1_mix_wikitext_alpaca_1bit_BitDistiller_baseline"

# Download the entire model into the current directory
snapshot_download(repo_id=model_name, local_dir="models/TinyLlama_v1.1_mix_wikitext_alpaca_1bit_BitDistiller_baseline")

print(f"Model {model_name} downloaded to TinyLlama_v1.1_mix_wikitext_alpaca_2bit_BitDistiller_baseline")
