#!/bin/bash

# Script to mix JSON datasets with predefined arguments (minimal checks)

# Define the arguments directly in the script
SMALL_DATASET_PATH="../datasets/tinyllama_v1.1/wikitext_T0.7_N1024_S42_3000.json"
LARGE_DATASET_PATH="../datasets/tinyllama_v1.1/alpaca_T0.7_N1024_S42_5000.json"
MIXING_RATIO="3:5"
SMALL_OVERSAMPLE="2"
OUTPUT_PATH="../datasets/tinyllama_v1.1/my_mixed_data.json"

# Run the Python script with the predefined arguments
python "mix_data.py" "$SMALL_DATASET_PATH" "$LARGE_DATASET_PATH" "$MIXING_RATIO" "$SMALL_OVERSAMPLE" "$OUTPUT_PATH"

