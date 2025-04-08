export MODEL_PATH='../models/Llama_3bit_OG/'
export SAVE_PATH=$2
export MASTER_ADDR="localhost"
export MASTER_PORT="1321"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"

deepspeed --num_gpus=1 train.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $1 \
    --model_max_length 1024 \
    --output_dir $SAVE_PATH \
    --logging_dir $3 \
    --num_train_epochs $4 \
    --bf16 True \
    --seed 42 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --evaluation_strategy "steps" \
    --eval_steps 40 \
    --load_best_model_at_end True \
    --save_strategy "steps" \
    --save_steps 40 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --lr_scheduler_type "constant" \
    --weight_decay 0. \
    --logging_steps 1 \
    --deepspeed config/zero.json \
    --bits 1 \
    --quant_type int1-asym \
    --q_group_size 128 \
    --train_kd True \
    --kd_loss_type "cakld" \
    --max_train_samples 999999 \
    --clip ../quantization/clip_cache/Llama_3bit_OG/int1-g128.pt
