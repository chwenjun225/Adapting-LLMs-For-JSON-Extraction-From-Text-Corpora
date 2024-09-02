# Compare LORA and QLORA on Alpaca dataset with same effective batch size ~32, lr sched, and lr.
# Reference for some hyperparams: https://arxiv.org/abs/2305.14314
# LORA (pure bf16)
# https://wandb.ai/answerdotai/fsdp/runs/gb34o6p4?workspace=user-k-answer-ai
# NOTE: Loss curve is flat - 1) use lower lr ? 2) start immediate annealing get_cosine_one_cycle_scheduler(..., min_lr_fraction=0.0)

# inference.sh 
lm_eval --model hf \
--model_args pretrained=chwenjun225/lora_adapters,load_in_4bit=True,parallelize=True \
--tasks lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa \
--device cuda \
--batch_size auto

# Finetune_Llama2_7b_Epoch5_LenSeq4096_QLORA_BF16_BatchSize2_GradAccum4_InsuranceBrands
python ./Codes/FSDP_QLORA/train.py \
--project_name Finetune_Llama2_7b_Epoch5_LenSeq4096_QLORA_BF16_BatchSize2_GradAccum4_InsuranceBrands \
--num_epochs 5 \
--batch_size 2 \
--context_length 4096 \
--train_type qlora \
--precision bf16 \
--use_gradient_checkpointing true \
--use_cpu_offload true \
--dataset insurance_brands \
--reentrant_checkpointing true \
--save_model True \
--output_dir ./Results \
--model_name meta-llama/Llama-2-7b-hf \
--log_to wandb \
--gradient_accumulation_steps 4 

# LoRA bf16
python train.py \
--model_name meta-llama/Llama-2-7b-hf \
--gradient_accumulation_steps 2 \
--batch_size 8 \
--context_length 512 \
--num_epochs 1 \
--train_type lora \
--use_gradient_checkpointing False \
--use_cpu_offload False \
--log_to wandb \
--dataset alpaca \
--verbose false \
--save_model true \
--output_dir ~/models/lora_alpaca

# QLORA (pure bf16)
python train.py \
--model_name meta-llama/Llama-2-7b-hf \
--gradient_accumulation_steps 2 \
--batch_size 8 \
--context_length 512 \
--num_epochs 1 \
--train_type qlora \
--use_gradient_checkpointing False \
--use_cpu_offload False \
--log_to wandb \
--dataset alpaca \
--verbose false \
--save_model true \
--output_dir ~/models/qlora_alpaca

# QLORA (autocast bf16)
python train.py \
--model_name meta-llama/Llama-2-7b-hf \
--precision bf16_buffers_autocast \
--gradient_accumulation_steps 2 \
--batch_size 8 \
--context_length 512 \
--num_epochs 1 \
--train_type qlora \
--use_gradient_checkpointing False \
--use_cpu_offload False \
--log_to wandb \
--dataset alpaca \
--verbose false \
--save_model true \
--output_dir ~/models/qlora_alpaca_autocast_buffers_bf16

# stop instance
# requires: az login --use-device-code
az vm deallocate -g resource-group-us-east -n a100-duo

export CUDA_VISIBLE_DEVICES=3,4
python train.py \
--world_size 2 \
--model_name meta-llama/Llama-2-7b-hf \
--gradient_accumulation_steps 2 \
--batch_size 1 \
--context_length 512 \
--num_epochs 1 \
--sharding_strategy full_shard \
--precision bf16 \
--train_type hqq_lora \
--use_gradient_checkpointing true \
--use_cpu_offload false \
--log_to stdout \
--dataset alpaca \
--verbose true

export CUDA_VISIBLE_DEVICES=4,5
python train.py \
--world_size 2 \
--master_port 12356 \
--model_name meta-llama/Llama-2-7b-hf \
--gradient_accumulation_steps 2 \
--batch_size 1 \
--context_length 512 \
--num_epochs 1 \
--sharding_strategy full_shard \
--precision bf16 \
--train_type hqq_lora \
--use_gradient_checkpointing true \
--use_cpu_offload false \
--log_to stdout \
--dataset dummy \
--verbose true

export CUDA_VISIBLE_DEVICES=3,4
python train.py \
--world_size 3 \
--model_name meta-llama/Llama-2-70b-hf \
--gradient_accumulation_steps 2 \
--batch_size 1 \
--context_length 4096 \
--num_epochs 1 \
--sharding_strategy full_shard \
--precision bf16 \
--train_type hqq_dora \
--use_gradient_checkpointing true \
--use_cpu_offload false \
--log_to wandb \
--dataset dummy \
--verbose true      