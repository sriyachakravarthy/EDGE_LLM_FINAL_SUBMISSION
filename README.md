# EDGE LLM FINAL SUBMISSION
# Introduction
 
This repository has preliminary submission details for [**Edge-Device Large Language Model Competition**](https://github.com/TianjinYellow/EdgeDeviceLLMCompetition-Starting-Kit).
## Name of the team and team members 
### Byte Crunchers
- Prof. Chetan Singh Thakur
- Madhuvanthi Srivatsav R 
- Arun C R 
- Sriya R G  
- Dhruva Kashyap
 
## Track chosen: Compression challenge
### Strategy:

1. For pruning qwen and llama models, we utilized [Shortened LLaMA: Depth Pruning for Large Language Models with Comparison of Retraining Methods](https://arxiv.org/abs/2402.02834) which adopts depth pruning that removes entire layers or blocks, while keeping the size of the remaining weights unchanged.
2. For finetuning, as mentioned in the starter kit of the competition, we used [**c4 dataset**] (https://huggingface.co/datasets/c4) and [**alpaca dataset**] (https://huggingface.co/datasets/silk-road/alpaca-data-gpt4-chinese) 
 

### Evaluating local models

1. Download models from Original_models folder in fp16 format using the provided link in ``saved_models.txt`` into a folder.
2. The downloaded folders' content should look like
   ```
   ├── config.json # Configuration file for the model
   ├── generation_config.json # Generation-specific configuration
   ├── special_tokens_map.json # Mapping of special tokens
   ├── tokenizer_config.json # Configuration file for the tokenizer
   ├── tokenizer.json # Tokenizer data file
   ├── model.safetensors.index.json # Index file for the model weights
   ├── model-00001-of-00002.safetensors # Part 1 of model weights in Safetensors format
   └── model-00002-of-00002.safetensors # Part 2 of model weights in Safetensors format
   ```
                   
#### Running the tasks
 
Tasks on the downloaded models should be run using the following code

```
# remove -r latest if reusing previous examples is not intended
opencompass --datasets [name of the dataset] --hf-type chat (base for phi2) \
--models path_to_the downloaded_model_folder --debug \
--model-kwargs device_map='auto' trust_remote_code=True \
--batch-size 1 -r latest --max-out-len 1024 --max-num-workers 1
```

#### Commands we used

##### PHI
```
opencompass --datasets truthfulqa_gen commonsenseqa_7shot_cot_gen_734a22 gsm8k_gen humaneval_gen FewCLUE_chid_gen bbh_gen    --hf-type base --hf-path path_to_phi_model  --tokenizer-path microsoft/phi-2  --model-kwargs device_map='auto' trust_remote_code=True torch_dtype=torch.float16   --max-num-workers 4   --max-out-len 1024   -r latest   --batch-size 4 \
```
##### QWEN
```
opencompass --datasets truthfulqa_gen commonsenseqa_7shot_cot_gen_734a22 gsm8k_gen humaneval_gen FewCLUE_chid_gen bbh_gen    --hf-type chat   --hf-path path_to_qwen_15_model   --tokenizer-path Qwen/Qwen2-7B-Instruct  --model-kwargs device_map='auto' trust_remote_code=True torch_dtype=torch.float16   --max-num-workers 4   --max-out-len 1024   -r latest   --batch-size 4 \
```
##### LLAMA
```
opencompass --datasets truthfulqa_gen commonsenseqa_7shot_cot_gen_734a22 gsm8k_gen humaneval_gen FewCLUE_chid_gen bbh_gen    --hf-type chat   --hf-path path_to_llama_20_model   --tokenizer-path meta-llama/Meta-Llama-3.1-8B-Instruct   --model-kwargs device_map='auto' trust_remote_code=True torch_dtype=torch.float16   --max-num-workers 4   --max-out-len 1024   -r latest   --batch-size 4 \
```


 ### Checking peak memory usage during inference
```
python peak_memory_usage.py --path_to_model_folder
```
 
# Submissions
1. Submitted models can be found in the link provided in ``saved_model.txt``


## Results 

### Evaluation Results Summary 



| Model       | Commonsenseqa_gen | FewChid_gen | bbh_gen | HumanEval | GSM8K | TruthfulQA | Memory       | Throughput |
|-------------|-------------------|-------------|---------|-----------|-------|------------|--------------|------------|
| **Llama3**  | 20                | 17.2        | 1.05    | 7.01      | 1.22  | 2.20       | 8.584 GB     |            |
| **Qwen**    | 0                 | 0.55        | 1.84    | 0         | 0.83  | 0          | 7 GB         |            |
| **Phi**     |                   |             |         |           |       |            |              |            |


#### Notes

- hftype chat was used while generating results with batch size 1 and max output length 1024 for llama and qwen models and base type was used for phi2


#### System Configuration

- **CPU Configuration**: 2P 96C Genoa AMD CPU
- **System RAM**: 1.5 TB
- **System Storage**: 48 TB
- **GPU Configuration**: 4 x AMD MI210 (64 GB)
- **Operating System**: Ubuntu 22.04 LTS
- **ROCm Version**: 5.7

-The Throughput and memory evaluation numbers from obtained from Nvidia A100 GPU with the following specs:

- CPU Configuration: AMD EPYC 7742 64-Core Processor    
- System RAM: 512 GB
- GPU Configuration: 4 x Nvidia A100 (40 GB)
- Operating System: Ubuntu 22.04.4 LTS
- CUDA Version: 12.4

### MLC compilation

- Link for converted_model file can be found in converted_weight/converted_weight.txt
- Optionally, it can be found in converted_model folder
- As per the feedback, APK file,  script to package the MLC model file into the APK and a screenshot of the successful run have been submitted in this repository.
- Refer to the Submission Commands.txt
  


