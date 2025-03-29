import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from safetensors.torch import load_file as load_safetensors
from huggingface_hub import snapshot_download
from scipy.stats import entropy
from transformers import AutoModelForCausalLM
from weight_distribution import plot_histogram, plot_overlaid_histograms
import csv
import re

def compute_entropy(weights):
    flat = weights.detach().cpu().to(torch.float32).numpy().flatten()
    hist, bin_edges = np.histogram(flat, bins=256, density=True)
    return entropy(hist + 1e-8)

def compute_frobenius_norm(w1, w2, device):
    return torch.norm(w1.to(device).to(torch.float32) - w2.to(device).to(torch.float32), p='fro').item()

def compute_svd_spectrum(weights, device):
    shape = weights.shape
    if len(shape) == 2:
        w32 = weights.detach().to(device).to(torch.float32)
        u, s, v = torch.linalg.svd(w32)
        return s.cpu().numpy()
    return None


def load_model_weights(model_dir):
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        return model.state_dict()
    except Exception as e:
        print("Failed to load model from Hugging Face Transformers:", e)
        return {}

# def load_model_weights(model_dir):
#     weight_files = [f for f in os.listdir(model_dir)
#                     if f.endswith(('.safetensors', '.pt')) and 'model' in f]
#     state_dict = {}
#     for f in weight_files:
#         path = os.path.join(model_dir, f)
#         if f.endswith(".safetensors"):
#             state_dict.update(load_safetensors(path))
#         elif f.endswith(".pt"):
#             state_dict.update(torch.load(path, map_location="cpu"))
#     return state_dict

IMPORTANT_LAYERS = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]


def extract_block_number(key):
    match = re.search(r'layers\.(\d+)', key)
    return int(match.group(1)) if match else -1

def analyze_weights(model1_dict, model2_dict=None, layers=None, device="cpu", last_n=None):
    selected_layers = layers if layers else IMPORTANT_LAYERS
    results = []

    matching_keys = [name for name in model1_dict.keys() if any(layer in name for layer in selected_layers)]
    matching_keys.sort(key=extract_block_number)

    if last_n is not None and last_n > 0:
        matching_keys = matching_keys[-last_n:]

    for name in matching_keys:
        w1 = model1_dict[name]
        safe_name = name.replace(".", "_").replace("/", "_")
        print(f"\nAnalyzing: {name}")
        print(f" - Shape: {tuple(w1.shape)}")

        H1 = compute_entropy(w1)
        print(f" - Entropy (model1): {H1:.4f}")

        s1 = compute_svd_spectrum(w1, device)
        if s1 is not None:
            print(f" - Top singular values (model1): {s1[:5]}")

        w1_sample = w1.detach().to(torch.float32).cpu().flatten()
        if w1_sample.numel() > 1_000_000:
            idx = torch.randperm(w1_sample.numel())[:1_000_000]
            w1_sample = w1_sample[idx]
        plot_histogram(w1_sample.numpy(), f"{safe_name}_model1_{H1:.4f}", "histograms")

        H2, s2, norm_diff = None, None, None

        if model2_dict and name in model2_dict:
            w2 = model2_dict[name]
            H2 = compute_entropy(w2)
            print(f" - Entropy (model2): {H2:.4f}")
            print(f" - Delta Entropy: {(H1 - H2):.4f}")

            s2 = compute_svd_spectrum(w2, device)
            if s2 is not None:
                delta_s = s1[:5] - s2[:5]
                print(f" - Top singular values (model2): {s2[:5]}")
                print(f" - Delta SVD: {delta_s}")

            norm_diff = compute_frobenius_norm(w1, w2, device)
            print(f" - Frobenius norm difference: {norm_diff:.4f}")

            w2_sample = w2.detach().to(torch.float32).cpu().flatten()
            if w2_sample.numel() > 1_000_000:
                idx = torch.randperm(w2_sample.numel())[:1_000_000]
                w2_sample = w2_sample[idx]

            plot_histogram(w2_sample.numpy(), f"{safe_name}_model2_{H2:.4f}", "histograms")
            plot_overlaid_histograms(w1_sample.numpy(), w2_sample.numpy(),
                                     f"{safe_name}_overlay_{H1:.4f}_{H2:.4f}", "histograms")

        results.append({
            'layer': name,
            'entropy_model1': round(H1, 4),
            'entropy_model2': round(H2, 4) if H2 is not None else None,
            'delta_entropy': round(H1 - H2, 4) if H2 is not None else None,
            'frobenius_norm_diff': round(norm_diff, 4) if norm_diff is not None else None,
            'svd_model1': s1[:5].tolist() if s1 is not None else [],
            'svd_model2': s2[:5].tolist() if s2 is not None else [],
            'delta_svd': (s1[:5] - s2[:5]).tolist() if (s1 is not None and s2 is not None) else []
        })

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--compare_id', type=str, default=None)
    parser.add_argument('--layers', nargs='*', default=None)
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--last_n', type=int, default=None, help="Only analyze the last N matching layers")
    args = parser.parse_args()

    print(f"Downloading model: {args.model_id}")
    model1_dir = snapshot_download(args.model_id)
    model1_dict = load_model_weights(model1_dir)

    model2_dict = None
    if args.compare_id:
        print(f"Downloading comparison model: {args.compare_id}")
        model2_dir = snapshot_download(args.compare_id)
        model2_dict = load_model_weights(model2_dir)

    print(f"Using device: {args.device}")
    os.makedirs("histograms", exist_ok=True)
    csv_path = os.path.join("histograms", "layerwise_analysis.csv")

    results = analyze_weights(model1_dict, model2_dict, args.layers, args.device, args.last_n)
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = [
            "layer",
            "entropy_model1",
            "entropy_model2",
            "delta_entropy",
            "frobenius_norm_diff",
            "svd_model1",
            "svd_model2",
            "delta_svd",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Saved CSV summary to {csv_path}")

if __name__ == '__main__':
    #python weight_analysis.py --model_id Heisenger/Llama-2-7b-hf_1bit_int --compare_id meta-llama/Llama-2-7b-hf --last_n 10
    main()
