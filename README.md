# 2024-09-02:16:22:58,349 INFO     
[evaluation_tracker.py:269] Output path not provided, skipping saving results aggregated
hf (pretrained=chwenjun225/lora_adapters,load_in_4bit=True,parallelize=True), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: auto (32)
|    Tasks     |Version|Filter|n-shot|  Metric  |   |Value |   |Stderr|
|--------------|------:|------|-----:|----------|---|-----:|---|-----:|
|arc_challenge |      1|none  |     0|acc       |↑  |0.4258|±  |0.0144|
|              |       |none  |     0|acc_norm  |↑  |0.4556|±  |0.0146|
|arc_easy      |      1|none  |     0|acc       |↑  |0.7445|±  |0.0089|
|              |       |none  |     0|acc_norm  |↑  |0.7235|±  |0.0092|
|hellaswag     |      1|none  |     0|acc       |↑  |0.5674|±  |0.0049|
|              |       |none  |     0|acc_norm  |↑  |0.7531|±  |0.0043|
|lambada_openai|      1|none  |     0|acc       |↑  |0.7206|±  |0.0063|
|              |       |none  |     0|perplexity|↓  |3.6835|±  |0.0750|
|openbookqa    |      1|none  |     0|acc       |↑  |0.3260|±  |0.0210|
|              |       |none  |     0|acc_norm  |↑  |0.4420|±  |0.0222|
|piqa          |      1|none  |     0|acc       |↑  |0.7818|±  |0.0096|
|              |       |none  |     0|acc_norm  |↑  |0.7802|±  |0.0097|
|winogrande    |      1|none  |     0|acc       |↑  |0.6867|±  |0.0130|



















































<!-- 
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
``` -->
