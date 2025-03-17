MODEL_PATH=$1
QUANT_TYPE=$2
BITS=$3

python wiki_ppl.py --model $MODEL_PATH --quant_type $QUANT_TYPE --bits $BITS --group_size 128

# TODO: add back hellaswag when needed
CUDA_VISIBLE_DEVICES=0 python llm_eval.py --model $MODEL_PATH --eval_tasks arc_easy,arc_challenge,winogrande,piqa --test_set --bits $BITS --group_size 128 --quant_type $QUANT_TYPE --num_fewshot 0 

# TODO: add back MMLU when needed as it is expensive
# CUDA_VISIBLE_DEVICES=0 python llm_eval.py --model  ../../train/ckpts/tinyllama_v1.1/int2-g128/checkpoint-12/ --eval_tasks hendrycksTest-* --test_set --bits 2 --group_size 128 --quant_type int --num_fewshot 5
