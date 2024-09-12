# 2024-09-02:16:22:58,349 INFO - meta-llama/Llama-2-7b-hf
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

## Inference
### Instruction: 
Extract information that you have learned from this source text:  
MUSIC
Pucker up! Kiss to open final 'End of the Road' tour in Cincinnati 💋
Portrait of Luann GibbsLuann Gibbs
Cincinnati Enquirer

The final leg of the Kiss "End of the Road" tour begins in Cincinnati. The iconic band are wrapping up a 50-year career with a North American tour that starts at Heritage Bank Center in Cincinnati, and ends at New York City's Madison Square Garden. Tickets go on sale Friday, June 9, 2023.
The end of the road begins in Cincinnati. The legendary rock 'n' roll band Kiss is closing out a 50-year career, but before the band packs away its iconic makeup and wild costumes, the boys are taking one last ride around the world with a final tour, fittingly titled the "End of the Road" tour. It will span 50 dates around the world, and the North American leg kicks off Oct. 19 right here in Cincinnati.

Tickets go on sale Friday, June 9, for the show, which will take place at Heritage Bank Center (100 Broadway, Downtown). The tour wraps up in December with a massive final show at Madison Square Garden in New York City.

Concert dates:Cincinnati's full 2023 concert calendar 🎵

Kiss was formed in New York City in 1973 by members Paul Stanley, Gene Simmons, Ace Frehley and Peter Criss. With greasepaint makeup and outrageous costumes, the bandmembers took on the personae of comic book-style characters, and their "shock-rock" style live performances have been known to feature fire-breathing, blood-spitting, levitating drum kits and pyrotechnics. Considered one of the most influential rock bands of all time and one of the best-selling bands of all time, Kiss has sold more than 75 million records worldwide, earned 30 gold albums, and all four original members have been inducted into the Rock and Roll Hall of Fame.

The current lineup includes Stanley, Simmons, guitarist Tommy Thayer and drummer Eric Singer.

Need a break? Play the USA TODAY Daily Crossword Puzzle.

Kiss 2023 North American End of the Road tour dates:
Oct. 19: Cincinnati, Heritage Bank Center
Oct. 20: Detroit, Little Caesars Arena
Oct. 22: Cleveland, Rocket Mortgage FieldHouse
Oct. 23: Nashville, Bridgestone Arena
Oct. 25: St. Louis, Enterprise Center
Oct. 27: Fort Worth, Texas, Dickies Arena           
Oct. 29: Austin, Moody Center
Nov. 1: Palm Springs, Calif. Acrisure Arena
Nov. 3: Los Angeles, Hollywood Bowl
Nov. 6: Seattle, Climate Pledge Arena
Nov. 8: Vancouver, Rogers Arena
Nov. 10: Edmonton, Alberta, Rogers Place
Nov. 12: Calgary, Alberta, Scotiabank Saddledome
Nov. 13: Saskatoon, Saskatchewan, SaskTel Centre
Nov. 15: Winnipeg, Manitoba, Canada Life Centre
Nov. 18: Montreal, Quebec, Centre Bell
Nov. 19: Quebec, Videotron Centre
Nov. 21: Ottawa, Ontario, Canadian Tire Centre
Nov. 22: Toronto, Ontario, Scotiabank Arena
Nov. 24: Knoxville, Tenn., Thompson-Boling Arena
Nov. 25: Indianapolis, Gainbridge Fieldhouse
Nov. 27: Rosemont, Illinois, Allstate Arena
Nov. 29: Baltimore, CFG Bank Arena
Dec. 1: New York City, Madison Square Garden
Dec. 2: New York City, Madison Square Garden

### Response:
```
{

    'Country': 'United States', 

    'Headline': 'Pucker up! Kiss to open final 'End of the Road' tour in Cincinnati 💋', 

    'Hit Sentence': '... Stanley, Simmons, guitarist Tommy Thayer and drummer Eric Singer. Need a break? Play the USA TODAY Daily Crossword Puzzle. Kiss 2023 ...', 

    'Influencer': 'Luann Gibbs', 

    'Language': 'English', 

    'Opening Text': 'The final leg of the Kiss "End of the Road" tour begins in Cincinnati. The iconic band are wrapping up a 50-year career with a North American tour ...', 

    'Source': 'Cincinnati Enquirer', 
    
    'URL': 'https://www.cincinnati.com/story/entertainment/music/2023/06/07/kiss-end-of-the-road-tour-cincinnati/71228854007/'
    
    }
```
</s>

</mark>



















































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
