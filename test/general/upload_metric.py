from pytablewriter import LatexTableWriter
import json
import argparse

from huggingface_hub import HfApi

parser = argparse.ArgumentParser()
parser.add_argument("--metrics_file", type=str, help="json file with model metrics")
parser.add_argument("--model_repo", type=str, "model repo to upload metric to")

args = parser.parse_args()

table = LatexTableWriter()
table.headers = ["PPL", "arc_easy", "arc_challenge", "piqa", "winogrande", "hellaswag", "mmlu"]

results_json = json.load(args.metrics_file)
results = []
for metric in table.headers:
    if metric in results_json:
        if "acc" in results_json[metric]:
            acc, std = results_json[metric]['acc'], results_json[metric]['std']
            acc *= 100
            std *= 100

            results.append(f"{acc:.2f}+-{std:.2f}")
        else:
            results.append(f"{100 * results_json[metric]:.2f}")
    else:
        results.append("")


table.value_matrix = [results]

# TODO: consider making this a temporary file, uploading within here
temp_file_path = "README.md"
with open(temp_file_path, "w+ ") as f:
    f.write(
        "**Metrics Table**:\n"
        "$$\n"
        f"{table.write_table()}"
        "$$\n"
        )

# optionally add the metrics to the yaml data too

api = HfApi()
api.upload_file(
    path_or_fileobj=temp_file_path,
    path_in_repo="README.md",
    repo_id=args.model_repo
)

# TODO: remove file