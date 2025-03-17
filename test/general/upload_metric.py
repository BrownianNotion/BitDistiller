from pytablewriter import MarkdownTableWriter
import json
import argparse
import pathlib

from huggingface_hub import HfApi, hf_hub_download 
from huggingface_hub.utils import EntryNotFoundError

parser = argparse.ArgumentParser()
parser.add_argument("--metrics_file", type=str, help="json file with model metrics")
parser.add_argument("--model_repo", type=str, help="model repo to upload metric to")

args = parser.parse_args()

table_writer = MarkdownTableWriter()
table_writer.headers = ["PPL", "arc_easy", "arc_challenge", "piqa", "winogrande", "hellaswag", "mmlu"]

with open(args.metrics_file, "r") as f:
    results_json = json.load(f)

results = []
for metric in table_writer.headers:
    if metric in results_json:
        # qa metrics
        if isinstance(results_json[metric], dict):
            acc, std = results_json[metric]['acc'], results_json[metric]['acc_stderr']
            acc *= 100
            std *= 100
            results.append(f"{acc:.2f} ± {std:.2f}")
        # ppl/mmlu
        else:
            scale_factor = 1 if metric == "PPL" else 100
            results.append(f"{scale_factor * results_json[metric]:.2f}")
    else:
        # blank (eg. mmlu/hellaswag may be blank because they take 
        # long to run and we can run later)
        results.append("-")

table_writer.value_matrix = [results]

modelcard_path = pathlib.Path(args.metrics_file).parent / "README.md"

with open(modelcard_path, "w+") as f:
    f.write(
        "**Metrics Table**:\n"
        )
    table_writer.stream = f
    table_writer.write_table()

api = HfApi()
api.upload_file(
    path_or_fileobj=modelcard_path,
    path_in_repo="README.md",
    repo_id=args.model_repo,
    commit_message="Upload model metrics"
)