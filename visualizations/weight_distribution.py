import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def plot_overlaid_histograms(data1, data2, title, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))

    # Compute bins separately for each histogram
    hist1, bins1 = np.histogram(data1, bins=100)
    hist2, bins2 = np.histogram(data2, bins=100)

    # Use bin centers
    centers1 = (bins1[:-1] + bins1[1:]) / 2
    centers2 = (bins2[:-1] + bins2[1:]) / 2

    plt.bar(centers1, hist1, width=(bins1[1] - bins1[0]), alpha=0.6, label="Quantized", color="blue", edgecolor="white", log=True)
    plt.bar(centers2, hist2, width=(bins2[1] - bins2[0]), alpha=0.6, label="Original", color="orange", edgecolor="white", log=True)

    plt.title(title)
    plt.xlabel("Weight Values")
    plt.ylabel("Log Count")
    plt.legend()
    plt.grid(True)


    plt.xlim(-0.5, 0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{title}.pdf"), format='pdf', dpi=150, bbox_inches='tight') 
    plt.close()
# def plot_overlaid_histograms(data1, data2, title, save_dir):
#     os.makedirs(save_dir, exist_ok=True)

#     plt.figure(figsize=(8, 5))
#     plt.hist(data1, bins=100, alpha=0.6, label="Quantized", color="blue", edgecolor="white", log=True)
#     plt.hist(data2, bins=100, alpha=0.6, label="Original", color="orange", edgecolor="white", log=True)
#     plt.title(title)
#     plt.xlabel("Weight Values")
#     plt.ylabel("Log Count")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, f"{title}.png")) 
#     plt.close()



def plot_histogram(data, title, save_path, max_points=1_000_000):
    data = data.flatten()
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif isinstance(data, list):
        data = np.array(data)

    if len(data) > max_points:
        idx = np.random.choice(len(data), size=max_points, replace=False)
        data = data[idx]

    hist, bins = np.histogram(data, bins=100)
    centers = (bins[:-1] + bins[1:]) / 2

    plt.figure(figsize=(8, 5))
    plt.bar(centers, hist, width=(bins[1] - bins[0]), alpha=0.7, edgecolor="white", log=True)
    plt.title(title)
    plt.xlabel("Weight Values")
    plt.grid(True)
    plt.xlim(-0.5, 0.5)  # Set x-axis limits
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=150, bbox_inches='tight')
    plt.close()
    

# def plot_histogram(data, title, save_path):
#     # plt.figure(figsize=(8, 5))
#     plt.hist(data, bins=100, alpha=0.7, log=True, edgecolor="white")
#     plt.title(title)
#     plt.xlabel("Weight Values")
#     plt.ylabel("Count")
#     plt.grid(True)
#     plt.savefig(save_path)
#     plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot weight distributions from a .pt file"
    )
    parser.add_argument(
        "--file_path", type=str, required=True, help="Path to the .pt file"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./plots", help="Directory to save the plots"
    )
    parser.add_argument(
        "--plot_name", type=str, required=True, help="Base name for the plot files"
    )
    args = parser.parse_args()

    # Load the .pt file
    file_path = args.file_path
    data = torch.load(file_path, map_location="cpu")  # Load onto CPU

    # Ensure save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    # Plot all weights histogram
    weights = []
    if "clip" in data and isinstance(data["clip"], list):
        for item in data["clip"]:
            if isinstance(item, tuple) and len(item) > 1:
                for tensor in item[1:]:
                    if isinstance(tensor, torch.Tensor):
                        tensor = tensor.to(dtype=torch.float32)
                        weights.append(tensor.view(-1).numpy())  # Flatten

    flattened_weights = np.concatenate(weights).tolist()
    plot_histogram(
        flattened_weights,
        "Histogram of All Extracted Tensor Values",
        os.path.join(args.save_dir, f"{args.plot_name}_all_weights.png"),
    )

    # Plot final down projection layer weight histogram
    down_proj_layer = data["clip"][-1]
    down_proj_weights = []
    for tensor in down_proj_layer[1:]:
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.to(dtype=torch.float32)
            down_proj_weights.append(tensor.view(-1).numpy())  # Flatten

    down_proj_weights = np.concatenate(down_proj_weights).tolist()
    plot_histogram(
        down_proj_weights,
        "Histogram of Final Down Projection Layer Weights",
        os.path.join(args.save_dir, f"{args.plot_name}_last_layer_weights.png"),
    )


if __name__ == "__main__":
    # Example usage:
    # python weight_distribution.py --file_path ../quantization/clip_cache/TinyLlama_v1.1/int2-g128.pt --plot_name tiny_llama_2_bit_128g
    main()
