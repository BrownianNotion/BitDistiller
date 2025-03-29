import os
import re
import pandas as pd
import matplotlib.pyplot as plt

def extract_layer_index(layer_name):
    match = re.search(r'layers\.(\d+)', layer_name)
    return int(match.group(1)) if match else None

def visualize_deltas(csv_paths, output_prefix="plot", title_suffix=""):
    all_dfs = []

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        df['layer_index'] = df['layer'].apply(extract_layer_index)
        df['source'] = os.path.basename(csv_path)
        all_dfs.append(df)

    full_df = pd.concat(all_dfs)

    # Plot delta entropy (dots only)
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, subdf in full_df.groupby("source"):
        ax.plot(
            subdf['layer_index'],
            subdf['delta_entropy'],
            marker='o',
            linestyle='None',
            alpha=0.7,
            label=key
        )
    ax.set_title(f"Layer-wise Delta Entropy {title_suffix}")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Δ Entropy")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_delta_entropy.png")
    plt.close()

    # Plot Frobenius norm difference (dots only)
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, subdf in full_df.groupby("source"):
        ax.plot(
            subdf['layer_index'],
            subdf['frobenius_norm_diff'],
            marker='o',
            linestyle='None',
            alpha=0.7,
            label=key
        )
    ax.set_title(f"Layer-wise Frobenius Norm Difference {title_suffix}")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Frobenius Norm Difference")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_frobenius_norm.png")
    plt.close()

    print("Plots saved to:", f"{output_prefix}_*.png")



visualize_deltas([                                                                                           
    "histograms/layerwise_analysis_1bit_fullprecision.csv",
    "histograms/layerwise_analysis_2bit_fullprecision.csv"
], output_prefix="comparison")