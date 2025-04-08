import json
import numpy as np
import pandas as pd
from collections import Counter
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from huggingface_hub import hf_hub_download


def extract_model_name(file_path):
    parts = file_path.split("/")
    if len(parts) > 2:
        return parts[-2]
    return None


def compute_perplexities(data, model, tokenizer, max_length=2048):
    perplexities = []
    for text in data:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        )
        input_ids = inputs["input_ids"].to(model.device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
            perplexities.append(perplexity)
    return perplexities


def analyze_distributions(file1, file2, eval_model_name="meta-llama/Llama-2-3b-hf", num_bootstrap=1000):
    def load_data(file_path):
        with open(file_path, "r") as f:
            data = [json.loads(line.strip()) for line in f]
        print(f"Loaded {len(data)} entries from {file_path}")
        filtered_data = [
            item[0][1] for item in data if isinstance(item, list) and len(item[0]) > 1
        ]
        print(f"Filtered {len(filtered_data)} valid entries from {file_path}")
        return filtered_data

    def compute_metrics(data):
        string_lengths = []
        unique_word_ratios = []
        repetition_rates = []
        non_char_ratios = []

        for text in data:
            words = text.split()
            string_lengths.append(len(text))
            unique_words = set(words)
            unique_word_ratios.append(len(unique_words) / len(words) if words else 0)
            word_counts = Counter(words)
            repeated_words = sum(count - 1 for count in word_counts.values() if count > 1)
            repetition_rates.append(repeated_words / len(words) if words else 0)
            non_chars = sum(1 for char in text if not char.isalnum() and not char.isspace())
            non_char_ratios.append(non_chars / len(text) if len(text) > 0 else 0)

        return {
            "string_lengths": string_lengths,
            "unique_word_ratios": unique_word_ratios,
            "repetition_rates": repetition_rates,
            "non_char_ratios": non_char_ratios,
        }

    def bootstrap(data, num_samples):
        means = []
        for _ in range(num_samples):
            sample = random.choices(data, k=len(data))
            means.append(np.mean(sample))
        return np.mean(means), np.std(means)

    # Load and process data
    data1 = load_data(file1)
    data2 = load_data(file2)
    if not data1 or not data2:
        print("One or both datasets are empty after filtering.")
        return None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(eval_model_name)
    # Step 1: Load raw config dict

    # # Manually download and patch config.json
    # config_path = hf_hub_download(repo_id=eval_model_name, filename="config.json")
    # with open(config_path, "r") as f:
    #     config_dict = json.load(f)

    # # Patch rope_scaling if needed
    # if "rope_scaling" in config_dict and "type" not in config_dict["rope_scaling"]:
    #     print("Patching rope_scaling for LLaMA 3 config...")
    #     config_dict["rope_scaling"] = {"type": "linear", "factor": 1.0}

    # # Now build a config object from the patched dict
    # config = AutoConfig.from_dict(config_dict)
    # # Step 2: Patch rope_scaling before instantiating
    # if "rope_scaling" in config_dict and "type" not in config_dict["rope_scaling"]:
    #     print("Patching rope_scaling for LLaMA 3 config...")
    #     config_dict["rope_scaling"] = {"type": "linear", "factor": 1.0}

    # # Step 3: Now instantiate the config safely
    # config = AutoConfig.from_dict(config_dict)
    # Load model with config
    model = AutoModelForCausalLM.from_pretrained(
        eval_model_name, torch_dtype=torch.float16, device_map="auto" # config=config,
    )
    model.eval()

    # Compute core metrics
    metrics1 = compute_metrics(data1)
    metrics2 = compute_metrics(data2)

    # Compute perplexity
    print(f"Computing perplexities using {eval_model_name}...")
    perplexities1 = compute_perplexities(data1, model, tokenizer)
    perplexities2 = compute_perplexities(data2, model, tokenizer)

    # Add perplexity to metrics
    metrics1["perplexity"] = perplexities1
    metrics2["perplexity"] = perplexities2

    # Bootstrap summary
    results = []
    for metric_name in [
        "string_lengths",
        "unique_word_ratios",
        "repetition_rates",
        "non_char_ratios",
        "perplexity",
    ]:
        mean1, se1 = bootstrap(metrics1[metric_name], num_bootstrap)
        mean2, se2 = bootstrap(metrics2[metric_name], num_bootstrap)
        diff_mean = mean1 - mean2
        results.append(
            {
                "Metric": metric_name.replace("_", " ").title(),
                extract_model_name(file1): f"{mean1:.3f} ± {se1:.3f}",
                extract_model_name(file2): f"{mean2:.3f} ± {se2:.3f}",
                "Difference": f"{diff_mean:.3f}",
            }
        )

    return pd.DataFrame(results)


# Run analysis
if __name__ == "__main__":
    file1 = "data/datasets/Llama-2-7b-hf/mix_wiki_alpaca_8000.json"
    file2 = "data/datasets/Llama-3-3B/mix_wiki_alpaca_8000.json"
    eval_model = "meta-llama/Llama-2-7b-hf" #"meta-llama/Llama-3.2-3B"  # or try Llama-2-7b-hf if you have enough VRAM
    results = analyze_distributions(file1, file2, eval_model_name=eval_model)
    if results is not None:
        print(results.to_string(index=False))
