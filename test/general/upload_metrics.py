# script creates a modelcard (README.md) for the given repo and
# auto generates a markdown table on the README.md from the metrics file
from pytablewriter import MarkdownTableWriter
import json
import argparse
import pathlib

from huggingface_hub import HfApi, ModelCard, ModelCardData, create_repo

parser = argparse.ArgumentParser()
parser.add_argument("--metrics_json", type=str, help="json file with model metrics")
parser.add_argument("--model_id", type=str, help="name of hugging face model repo without username.")

args = parser.parse_args()

table_writer = MarkdownTableWriter()
table_writer.headers = ["PPL", "arc_easy", "arc_challenge", "piqa", "winogrande", "hellaswag", "mmlu", "QA Avg"]

with open(args.metrics_json, "r") as f:
    results_json = json.load(f)

    assert set(results_json.keys()).issubset(set(table_writer.headers))
    
    for metric in table_writer.headers:
        if metric in results_json and isinstance(results_json[metric], dict):
            assert set(["acc", "acc_stderr"]).issubset(set(results_json[metric].keys()))

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
        # blank (eg. mmlu may be blank because they take 
        # long to run and we can run later)
        results.append("-")

table_writer.value_matrix = [results]

card_data = ModelCardData(
    language='en',
    license='mit',
    base_model="TinyLlama/TinyLlama_v1.1",
)

results_summary = table_writer.dumps()

card = ModelCard.from_template(
    card_data=card_data,
    results_summary=results_summary,
    template_path="modelcard_template.md"
)

api = HfApi()
username = api.whoami()['name']
create_repo(repo_id=args.model_id, repo_type="model", exist_ok=True)
card.push_to_hub(repo_id=f"{username}/{args.model_id}", commit_message="Upload model metrics")
