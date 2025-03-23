rm -rf ckpts/dry_run logs/dry_run
bash train_dry_run.sh ../data/datasets/Llama-3.2-3B/mix_wiki_alpaca_64.json ./ckpts/dry_run ./logs/dry_run/ 1