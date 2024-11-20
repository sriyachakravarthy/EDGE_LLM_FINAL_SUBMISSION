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

   For pruning, we utilized [Shortened LLaMA: Depth Pruning for Large Language Models with Comparison of Retraining Methods](https://arxiv.org/abs/2402.02834) which adopts depth pruning that removes entire layers or blocks, while keeping the size of the remaining weights unchanged. 
 

### Evaluating local models

1. Download models in fp16 format using the provided link in ``saved_model.txt`` into a folder.
 
#### Running the tasks
 
Tasks on the downloaded models were run using the following code
```
# remove -r latest if reusing previous examples is not intended
opencompass --datasets [name of the dataset] --hf-type chat (base for phi2) \
--models path_to_the downloaded_model_folder --debug \
--model-kwargs device_map='auto' trust_remote_code=True \
--batch-size 1 -r latest --max-out-len 1024 --max-num-workers 1
```
 
 
# Submissions
1. Submitted models before mlc conpilation can be found in the link provided in ``saved_model.txt``

## Results 

### Evaluation Results Summary 


| Metric             | Result |
|--------------------|--------|
| CommonsenseQA      | 65.68 |
| BIG-Bench-Hard     | 59.4  |
| GSM8K              | 61.56 |
| HumanEval          | 31.1   |
| CHID               | 12.29  |
| TruthfulQA         | 0.18   |
| Throughput         | 34.66  inf/s    |
| Memory-Usage       | 6.534 GB      |

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
  


