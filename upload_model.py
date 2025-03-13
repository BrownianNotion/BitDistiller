from huggingface_hub import HfApi

api = HfApi()

username = "BrownianNotion"
api.upload_folder(
    folder_path="train/ckpts/tiny_llama_v1.1/int1-g128",
    repo_id=f"{username}/tinyllama_v1.1-int1-g128",
    repo_type="model",
    ignore_patterns=["**/checkpoint*"]
)