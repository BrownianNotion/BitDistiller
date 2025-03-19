import json
import random
import argparse

random.seed(42)


def mix_datasets(
    dataset_1_path="../datasets/tinyllama_v1.1/wikitext_T0.7_N1024_S42_3000.json",
    dataset_2_path="../datasets/tinyllama_v1.1/alpaca_T0.7_N1024_S42_5000.json",
    dataset_1_size=3000,
    dataset_2_size=5000,
    output_path="../datasets/tinyllama_v1.1/mix_wiki_alpaca_8000.json",
):
    """
    Mixes two JSON datasets
    """

    all_outputs = []

    with open(dataset_1_path, "r") as f:
        dataset_1 = [json.loads(line) for line in f]

    if dataset_1_size > len(dataset_1):
        print(
            f"Warning: dataset_1_size ({dataset_1_size}) is larger than the dataset_1 ({len(dataset_1)}). Taking all samples from dataset_1."
        )

    all_outputs.extend(random.sample(dataset_1, min(dataset_1_size, len(dataset_1))))

    with open(dataset_2_path, "r") as f:
        dataset_2 = [json.loads(line) for line in f]

    if dataset_2_size > len(dataset_2):
        print(
            f"Warning: dataset_2_size ({dataset_2_size}) is larger than the dataset_2 ({len(dataset_2)}). Taking all samples from dataset_2."
        )

    all_outputs.extend(random.sample(dataset_2, min(dataset_2_size, len(dataset_2))))

    random.shuffle(all_outputs)

    with open(output_path, "w") as f:
        for item in all_outputs:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Mix two JSON datasets.")
    parser.add_argument(
        "--dataset_1_path", type=str, required=True, help="Path to the first dataset."
    )
    parser.add_argument(
        "--dataset_2_path", type=str, required=True, help="Path to the second dataset."
    )
    parser.add_argument(
        "--dataset_1_size",
        type=int,
        required=True,
        help="Number of samples to take from the first dataset.",
    )
    parser.add_argument(
        "--dataset_2_size",
        type=int,
        required=True,
        help="Number of samples to take from the second dataset.",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save the mixed dataset."
    )

    args = parser.parse_args()

    mix_datasets(
        dataset_1_path=args.dataset_1_path,
        dataset_2_path=args.dataset_2_path,
        dataset_1_size=args.dataset_1_size,
        dataset_2_size=args.dataset_2_size,
        output_path=args.output_path,
    )
