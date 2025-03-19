import json
import random
import argparse
import numpy as np

random.seed(42)
np.random.seed(42)


def mix_datasets(
    small_dataset_path, large_dataset_path, mixing_ratio, small_oversample, output_path
):
    """
    Mixes two JSON datasets based on the specified mixing ratio and oversampling.
    If the large dataset is not large enough, it repeats the dataset.

    Args:
        small_dataset_path (str): Path to the smaller JSON dataset.
        large_dataset_path (str): Path to the larger JSON dataset.
        mixing_ratio (str): Mixing ratio in the format "small:large" (e.g., "3:5").
        small_oversample (int): Oversampling factor for the smaller dataset.
        output_path (str): Path to save the mixed JSON dataset.
    """

    try:
        small_ratio, large_ratio = map(int, mixing_ratio.split(":"))
    except ValueError:
        print("Invalid mixing ratio format. Use 'small:large' (e.g., '3:5').")
        return

    all_outputs = []

    # Load and oversample the smaller dataset
    with open(small_dataset_path, "r") as f:
        small_dataset = [json.loads(line) for line in f]

    for _ in range(small_oversample):  # Oversample
        all_outputs.extend(small_dataset)

    # Load and subsample/repeat the larger dataset
    with open(large_dataset_path, "r") as f:
        large_dataset = [json.loads(line) for line in f]

    small_dataset_size = len(small_dataset) * small_oversample
    large_dataset_size = int((large_ratio / small_ratio) * small_dataset_size)

    if large_dataset_size > len(large_dataset):
        print(
            f"Warning: Desired large dataset size ({large_dataset_size}) is larger than the actual size ({len(large_dataset)}). Repeating the large dataset."
        )

        repeated_large_dataset = []
        while len(repeated_large_dataset) < large_dataset_size:
            repeated_large_dataset.extend(large_dataset)

        all_outputs.extend(random.sample(repeated_large_dataset, large_dataset_size))

    else:
        all_outputs.extend(random.sample(large_dataset, large_dataset_size))

    random.shuffle(all_outputs)

    # Save the mixed dataset
    with open(output_path, "w") as f:
        for item in all_outputs:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mix two JSON datasets with specified ratios and oversampling."
    )
    parser.add_argument(
        "small_dataset_path", type=str, help="Path to the smaller JSON dataset."
    )
    parser.add_argument(
        "large_dataset_path", type=str, help="Path to the larger JSON dataset."
    )
    parser.add_argument("mixing_ratio", type=str, help="Mixing ratio (e.g., '3:5').")
    parser.add_argument(
        "small_oversample",
        type=int,
        help="Oversampling factor for the smaller dataset.",
    )
    parser.add_argument(
        "output_path", type=str, help="Path to save the mixed JSON dataset."
    )

    args = parser.parse_args()

    mix_datasets(
        args.small_dataset_path,
        args.large_dataset_path,
        args.mixing_ratio,
        args.small_oversample,
        args.output_path,
    )
