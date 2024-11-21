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
 To check the peak memory usage during inference, we ran ``peak_memory_usage.py``
```
python peak_memory_usage.py --model_name $path_to_model_folder
```
 #### Results
 #### Peak memory during inference for LLAMA
 <img width="297" alt="image" src="https://github.com/user-attachments/assets/5562227e-88d4-4833-93f2-150192a22b56">
 #### Peak memory during inference for QWEN
<img width="299" alt="image" src="https://github.com/user-attachments/assets/ef7e22a8-f40f-492a-9610-d15a423a5ba3">
 #### Peak memory during inference for PHI
 <img width="294" alt="image" src="https://github.com/user-attachments/assets/e8d756d3-ccc3-40d4-b2d0-a1ae5b790813">


 
# Submissions
1. Submitted models can be found in the link provided in ``saved_model.txt``
2. Original models (before MLC compilation) can be found in ``Original_Models`` folder from the above link
3. MLC Compiled models can be found in ``MLC_COMPILED_MODELS`` folder from the above link
4. Screenshots of the running app for phi2 has been provided in ``screenshots`` folder in this GitHub Repository


## Results 

### Evaluation Results Summary 


| Model       | Commonsenseqa_gen | FewChid_gen | bbh_gen | HumanEval | GSM8K | TruthfulQA | Memory       | Throughput             |
|-------------|-------------------|-------------|---------|-----------|-------|------------|--------------|----------------------- |
| **Llama3**  | 22.11             | 13.59       | 7.87    | 0.61      | 1.9   | 0.18       | 8.755 GB     | 71.61 inf/s 0.014 s    |
| **Qwen**    | 0                 | 0.55        | 1.84    | 0         | 0.83  | 0          | 7 GB         | 77.73 inf/s 0.0129 s   |
| **Phi**     | 65.68             | 12.29       | 59.4    | 31.1      | 61.56 | 0.18       |  6.143 GB    | 13.53 inf/s 0.0739 s   |

### Screenshots of the running app
![Application](https://github.com/user-attachments/assets/28a11291-c5af-46e9-a041-1ea4322b9042)
![Phi-2](https://github.com/user-attachments/assets/75ddd831-8068-4239-a062-64f11c5cb01e)

## Errors encountered
We encountered errors while compiling Llama and Qwen models. Refer to``` MLC COMPILATION ERRORS``` folder in this repository. We were able to compile all the models. However, Llama and Qwen fail at runtime due to out-of-memory issues on the target device. This comes despite our adherence to the competition rules, where the models in question utilize only around 9 GB of memory on the desktop.
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

- Link for converted_model file can be found in ``saved_models.txt`` 
- Refer to the Submission Commands.txt
  


