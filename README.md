# SynerGPUs - Optimize and combine vram of multiple GPUs 

陳文俊 - National Kaohsiung University of Science and Technology

This project provides scripts and tools to finetune large language models (LLMs) using multiple GPUs. The project leverages techniques such as Fully Sharded Data Parallel (FSDP) and Quantized Low-Rank Adaptation (QLoRA) to efficiently train models on large datasets.

## Features

- **Multi-GPU Training**: Utilize multiple GPUs to speed up the training process.
- **FSDP**: Fully Sharded Data Parallelism to optimize memory usage.
- **QLoRA**: Quantized Low-Rank Adaptation for efficient model finetuning.
- **Gradient Checkpointing**: Save memory by checkpointing intermediate activations.
- **CPU Offloading**: Offload computations to CPU to further optimize GPU memory usage.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```
    
2. Create and activate a conda environment:
    ```sh
    conda create -n fsdp_qlora_env python=3.10
    conda activate fsdp_qlora_env
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To finetune a model using the provided scripts, run the following command:

```sh
python ./Codes/FSDP_QLORA/train.py 
--model_name meta-llama/Llama-2-7b-hf 
--batch_size 2 
--context_length 128 
--precision bf16 
--train_type qlora 
--use_gradient_checkpointing true 
--use_cpu_offload true 
--dataset alpaca_sample 
--reentrant_checkpointing true 
--save_model True 
--output_dir ./Results
```
