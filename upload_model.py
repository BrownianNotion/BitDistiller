import argparse
from huggingface_hub import HfApi, create_repo, upload_large_folder

parser = argparse.ArgumentParser()
parser.add_argument("model_path", type=str, help="path to model checkpoint/folder, including tensorboard logs")
parser.add_argument("bits", type=int, help="num of bits the model is quantised to")
parser.add_argument("--quant_type", type=str, default="int", help="Quantisation method")
parser.add_argument("--extra_changes", type=str, default="", help="String summarising extra changes such as new losses/architecture. Eg. ce_loss")
parser.add_argument("--base_model", type=str, default="TinyLlama_v1.1", help="Base model that we quantised, defaults to TinyLlama_v1.1")
parser.add_argument("--overwrite", type=bool, default=False, 
        help="Whether to overwrite hugging face repo if model repo already exists. "
        "Specify ANY argument to set to True (argparse is weird in this way), don't specify for False."
        "Note, even if you accidentally ovewrite data with this option, you can revert the latest commit" \
        "on hugging face")

args = parser.parse_args()

model_name = f"{args.base_model}_{args.bits}bit_{args.quant_type}" 
if args.extra_changes:
    model_name += f"_{args.extra_changes}"

api = HfApi()
username = api.whoami()['name']

repo_id = f"{username}/{model_name}"
print(f"\033[32mUploading model to {repo_id} ...\033[0m") # weird chars at start/end make it print in green
create_repo(repo_id=model_name, repo_type="model", exist_ok=args.overwrite)
upload_large_folder(
   folder_path=args.model_path,
   repo_id=repo_id,
   repo_type="model"
)