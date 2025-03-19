#!/bin/bash

# Script to mix JSON datasets with predefined arguments

# Define the arguments directly in the script
DATASET_1_PATH="datasets/tinyllama_v1.1/wikitext_T0.7_N1024_S42_9939.json"
DATASET_2_PATH="datasets/tinyllama_v1.1/alpaca_T0.7_N1024_S42_20000.json"
DATASET_1_SIZE=6000
DATASET_2_SIZE=10000
OUTPUT_PATH="../datasets/tinyllama_v1.1/mix_wiki_alpaca_16000.json"

# Run the Python script with the required argument flags
python "mix_data.py" \
    --dataset_1_path "$DATASET_1_PATH" \
    --dataset_2_path "$DATASET_2_PATH" \
    --dataset_1_size "$DATASET_1_SIZE" \
    --dataset_2_size "$DATASET_2_SIZE" \
    --output_path "$OUTPUT_PATH"
