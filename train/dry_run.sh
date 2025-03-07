rm -rf ckpts/dry_run logs/dry_run
bash train_dry_run.sh ../data/datasets/tinyllama_v1.1/mix_wiki_alpaca_64.json ./ckpts/dry_run ./logs/dry_run/ 1