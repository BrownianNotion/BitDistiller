import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from safetensors.torch import load_file as load_safetensors
from huggingface_hub import snapshot_download
from scipy.stats import entropy
from weight_distribution import plot_histogram, plot_overlaid_histograms

def compute_entropy(weights):
    flat = weights.detach().cpu().to(torch.float32).numpy().flatten()
    hist, bin_edges = np.histogram(flat, bins=256, density=True)
    return entropy(hist + 1e-8)  # add small constant to avoid log(0)

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
    weight_files = [f for f in os.listdir(model_dir)
                    if f.endswith(('.safetensors', '.pt')) and 'model' in f]
    state_dict = {}
    for f in weight_files:
        path = os.path.join(model_dir, f)
        if f.endswith(".safetensors"):
            state_dict.update(load_safetensors(path))
        elif f.endswith(".pt"):
            state_dict.update(torch.load(path, map_location="cpu"))
    return state_dict

IMPORTANT_LAYERS = ["q_proj", "k_proj", "v_proj", "fc1", "fc2", "o_proj"]

import csv

def analyze_weights(model1_dict, model2_dict=None, layers=None, device="cpu"):
    """
    Compares weights from two models and returns a list of per-layer metrics.
    """
    selected_layers = layers if layers else IMPORTANT_LAYERS
    results = []

    for name, w1 in model1_dict.items():
        # Skip layers we don't want
        if not any(layer in name for layer in selected_layers):
            continue

        safe_name = name.replace(".", "_").replace("/", "_")
        print(f"\n🔍 Analyzing: {name}")
        print(f" - Shape: {tuple(w1.shape)}")

        # Compute metrics for model1
        H1 = compute_entropy(w1)
        print(f" - Entropy (model1): {H1:.4f}")

        s1 = compute_svd_spectrum(w1, device)
        if s1 is not None:
            print(f" - Top singular values (model1): {s1[:5]}")

        # Prepare histogram samples from model1
        w1_sample = w1.detach().to(torch.float32).cpu().flatten()
        if w1_sample.numel() > 1_000_000:
            idx = torch.randperm(w1_sample.numel())[:1_000_000]
            w1_sample = w1_sample[idx]
        # Save histogram for model1
        plot_histogram(w1_sample.numpy(), f"{safe_name}_model1_{H1:.4f}", "histograms")

        # Defaults for model2 metrics if not found
        H2, s2, norm_diff = None, None, None

        # If compare_id model exists and has this layer
        if model2_dict and name in model2_dict:
            w2 = model2_dict[name]
            H2 = compute_entropy(w2)
            print(f" - Entropy (model2): {H2:.4f}")
            print(f" - Δ Entropy: {(H1 - H2):.4f}")

            s2 = compute_svd_spectrum(w2, device)
            if s2 is not None:
                delta_s = s1[:5] - s2[:5]
                print(f" - Top singular values (model2): {s2[:5]}")
                print(f" - Δ SVD: {delta_s}")

            norm_diff = compute_frobenius_norm(w1, w2, device)
            print(f" - Frobenius norm difference: {norm_diff:.4f}")

            # Prepare histogram samples for model2
            w2_sample = w2.detach().to(torch.float32).cpu().flatten()
            if w2_sample.numel() > 1_000_000:
                idx = torch.randperm(w2_sample.numel())[:1_000_000]
                w2_sample = w2_sample[idx]
            # Save histogram for model2
            plot_histogram(w2_sample.numpy(), f"{safe_name}_model2_{H2:.4f}", "histograms")
            # Overlay histograms for side-by-side comparison
            plot_overlaid_histograms(w1_sample.numpy(), w2_sample.numpy(),
                                     f"{safe_name}_overlay_{H1:.4f}_{H2:.4f}", "histograms")

        # Append row to results
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
    args = parser.parse_args()

    print(f"⬇️ Downloading model: {args.model_id}")
    model1_dir = snapshot_download(args.model_id)
    model1_dict = load_model_weights(model1_dir)

    model2_dict = None
    if args.compare_id:
        print(f"⬇️ Downloading comparison model: {args.compare_id}")
        model2_dir = snapshot_download(args.compare_id)
        model2_dict = load_model_weights(model2_dir)

    print(f"🚀 Using device: {args.device}")

    # Ensure output dir exists
    os.makedirs("histograms", exist_ok=True)
    csv_path = os.path.join("histograms", "layerwise_analysis.csv")

    # Run analysis, write CSV
    results = analyze_weights(model1_dict, model2_dict, args.layers, args.device)
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

    print(f"📄 Saved CSV summary to {csv_path}")

if __name__ == '__main__':
    main()
