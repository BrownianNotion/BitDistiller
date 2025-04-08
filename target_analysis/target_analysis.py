import json
import numpy as np
import pandas as pd
from collections import Counter
import random


def extract_model_name(file_path):
    """
    Extract the middle part (model name) from the file path.

    Args:
        file_path (str): The file path string.

    Returns:
        str: The extracted model name.
    """
    parts = file_path.split("/")
    if len(parts) > 2:
        return parts[-2]  # Extract the second-to-last part
    return None


def analyze_distributions(file1, file2, num_bootstrap=1000):
    """
    Compare the distribution properties of two JSON files.
    Focuses on string-level variation, length, repetition, and non-character appearance.

    Args:
        file1 (str): Path to the first JSON file.
        file2 (str): Path to the second JSON file.
        num_bootstrap (int): Number of bootstrap samples.

    Returns:
        pd.DataFrame: A DataFrame containing the comparison results.
    """

    def load_data(file_path):
        with open(file_path, "r") as f:
            data = [json.loads(line.strip()) for line in f]  # Parse JSONL
        print(f"Loaded {len(data)} entries from {file_path}")
        # Ensure each item has at least two elements
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
            words = text.split()  # Split text into words by whitespace
            string_lengths.append(len(text))  # Length of the string

            # Unique word ratio
            unique_words = set(words)
            unique_word_ratios.append(len(unique_words) / len(words) if words else 0)

            # Repetition rate
            word_counts = Counter(words)
            repeated_words = sum(
                count - 1 for count in word_counts.values() if count > 1
            )
            repetition_rates.append(repeated_words / len(words) if words else 0)

            # Non-character ratio
            non_chars = sum(
                1 for char in text if not char.isalnum() and not char.isspace()
            )
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
        print(
            "One or both datasets are empty after filtering. Please check the input files."
        )
        return None

    # Compute metrics
    metrics1 = compute_metrics(data1)
    metrics2 = compute_metrics(data2)

    # Bootstrap to compute mean and standard error
    results = []
    for metric_name in [
        "string_lengths",
        "unique_word_ratios",
        "repetition_rates",
        "non_char_ratios",
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

    # Convert results to DataFrame
    comparison = pd.DataFrame(results)
    return comparison


# Example usage
if __name__ == "__main__":
    file1 = "data/datasets/Llama-2-7b-hf/mix_wiki_alpaca_8000.json"
    file2 = "data/datasets/Llama-3-3B/mix_wiki_alpaca_8000.json"  # "data/datasets/tinyllama_v1.1/mix_wiki_alpaca_8000.json"
    results = analyze_distributions(file1, file2)
    if results is not None:
        print(results.to_string(index=False))  # Print the DataFrame in a clean format
