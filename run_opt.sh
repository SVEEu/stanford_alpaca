set -x

pip install -r requirements.txt

export $MASTER_PORT=${$MASTER_PORT:-127.0.0.1}

torchrun --nproc_per_node=8 --master_port=$MASTER_PORT train.py \
    --cache_dir ./tmp/cache_dir \
    --config_name "facebook/opt-6.7b" \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir ./tmp/output_dir \
    --num_train_epochs 30 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'OPTDecoderLayer' \
    --tf32 True
