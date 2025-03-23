#!/bin/bash
rm -rf ckpts/dry_run logs/dry_run
bash train_dry_run.sh /workspace/BitDistiller/data/datasets/Llama-2-7b-hf/mix_wiki_alpaca_8000.json ./ckpts/dry_run ./logs/dry_run/ 1
