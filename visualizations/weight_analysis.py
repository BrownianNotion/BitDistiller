import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from safetensors.torch import load_file as load_safetensors
from huggingface_hub import snapshot_download
from scipy.stats import entropy
from weight_distribution import plot_histogram


def compute_entropy(weights):
    flat = weights.detach().cpu().numpy().flatten()
    hist, bin_edges = np.histogram(flat, bins=256, density=True)
    return entropy(hist + 1e-8)  # add small constant to avoid log(0)


def compute_frobenius_norm(w1, w2):
    return torch.norm(w1 - w2, p="fro").item()


def compute_svd_spectrum(weights):
    shape = weights.shape
    if len(shape) == 2:
        u, s, v = torch.linalg.svd(weights)
        return s.detach().cpu().numpy()
    return None


def load_model_weights(model_dir):
    weight_files = [
        f
        for f in os.listdir(model_dir)
        if f.endswith((".safetensors", ".pt")) and "model" in f
    ]
    state_dict = {}
    for f in weight_files:
        path = os.path.join(model_dir, f)
        if f.endswith(".safetensors"):
            state_dict.update(load_safetensors(path))
        elif f.endswith(".pt"):
            state_dict.update(torch.load(path, map_location="cpu"))
    return state_dict


def analyze_weights(model1_dict, model2_dict=None, layers=None):
    for name, w1 in model1_dict.items():
        if layers and not any(layer in name for layer in layers):
            continue

        print(f"\n🔍 Analyzing: {name}")
        print(f" - Shape: {tuple(w1.shape)}")

        H = compute_entropy(w1)
        print(f" - Entropy: {H:.4f}")

        s = compute_svd_spectrum(w1)
        if s is not None:
            print(f" - Top singular values: {s[:5]}")

        if model2_dict and name in model2_dict:
            w2 = model2_dict[name]
            norm_diff = compute_frobenius_norm(w1, w2)
            print(f" - Frobenius norm difference: {norm_diff:.4f}")

        # Histogram
        plot_weight_distribution(w1, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--compare_id", type=str, default=None)
    parser.add_argument("--layers", nargs="*", default=None)
    args = parser.parse_args()

    print(f"⬇️ Downloading model: {args.model_id}")
    model1_dir = snapshot_download(args.model_id)
    model1_dict = load_model_weights(model1_dir)

    model2_dict = None
    if args.compare_id:
        print(f"⬇️ Downloading comparison model: {args.compare_id}")
        model2_dir = snapshot_download(args.compare_id)
        model2_dict = load_model_weights(model2_dir)

    analyze_weights(model1_dict, model2_dict, args.layers)


if __name__ == "__main__":
    main()
