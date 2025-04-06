MODEL_PATH=$1
QUANT_TYPE=$2
BITS=$3
CLIP_PATH=$4

# python wiki_ppl.py --model $MODEL_PATH --quant_type $QUANT_TYPE --bits $BITS --group_size 128 --clipped_untrained 1 --clip_path $CLIP_PATH
python wiki_ppl.py --model $MODEL_PATH --quant_type $QUANT_TYPE --bits $BITS --group_size 128 --random 1

CUDA_VISIBLE_DEVICES=0 python llm_eval.py --model $MODEL_PATH --eval_tasks arc_easy,arc_challenge,winogrande,piqa,hellaswag --test_set --bits $BITS --group_size 128 --quant_type $QUANT_TYPE --num_fewshot 0 --random 1
# CUDA_VISIBLE_DEVICES=0 python llm_eval.py --model $MODEL_PATH --eval_tasks arc_easy,arc_challenge,winogrande,piqa,hellaswag --test_set --bits $BITS --group_size 128 --quant_type $QUANT_TYPE --num_fewshot 0 --clipped_untrained 1 --clip_path $CLIP_PATH 

# Uncomment for MMLU, not run by deafult as expensive 
# CUDA_VISIBLE_DEVICES=0 python llm_eval.py --model $MODEL_PATH --eval_tasks hendrycksTest-* --test_set --bits $BITS --group_size 128 --quant_type $QUANT_TYPE --num_fewshot 5
