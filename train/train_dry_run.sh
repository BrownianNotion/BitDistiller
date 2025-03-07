export MODEL_PATH='../models/TinyLlama_v1.1/'
export SAVE_PATH=$2
export MASTER_ADDR="localhost"
export MASTER_PORT="1321"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"
export WANDB_DISABLED=true  

deepspeed --num_gpus=1 train.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $1 \
    --model_max_length 1024 \
    --output_dir $SAVE_PATH \
    --logging_dir $3 \
    --num_train_epochs $4 \
    --bf16 True \
    --seed 42 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --evaluation_strategy "steps" \
    --eval_steps 4 \
    --load_best_model_at_end True \
    --save_strategy "steps" \
    --save_steps 20 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --lr_scheduler_type "constant" \
    --weight_decay 0. \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --deepspeed config/zero.json \
    --bits 2 \
    --quant_type int2-asym \
    --q_group_size 128 \
    --train_kd True \
    --kd_loss_type "cakld" \
    --max_train_samples 999999 \
    --clip ../quantization/clip_cache/TinyLlama_v1.1/int2-g128.pt
