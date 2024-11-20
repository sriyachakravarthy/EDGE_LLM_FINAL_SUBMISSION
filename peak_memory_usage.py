import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import gc
import argparse

# Function to check RAM usage (in GB)
def get_ram_usage():
    process = psutil.Process()
    memory_info = process.memory_info()  # Memory usage in bytes
    return memory_info.rss / 1024**3  # Return in GB

# Set up argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a model and track memory usage.")
    parser.add_argument("model_name", type=str, help="Path to the model directory")
    return parser.parse_args()

# Main function
def main():
    # Parse command-line arguments
    args = parse_args()
    model_name = args.model_name  # Get model name from argument
    print('Running results on', model_name)
    
    # Load the model and tokenizer in FP16 to optimize memory usage
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))
    
    # Example input with a sequence length of 2000
    sequence_length = 2000
    batch_size = 1
    input_text = "This is a sample input for the model. " * (sequence_length // 6)  # Adjust the input length to simulate 2000 tokens
    
    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=sequence_length)
    inputs = {key: val.to("cuda") for key, val in inputs.items()}  # Move to GPU
    
    # Run inference and track memory usage
    start_time = time.time()
    
    # Get memory usage before inference
    ram_before_inference = get_ram_usage()
    print(f"RAM usage before inference: {ram_before_inference:.2f} GB")
    
    # Get initial peak GPU memory usage before inference
    torch.cuda.reset_peak_memory_stats()
    peak_memory_before = torch.cuda.max_memory_allocated()
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Track memory usage during inference
    torch.cuda.synchronize()  # Make sure memory stats are updated
    
    # Get peak GPU memory usage during inference
    peak_memory_during_inference = torch.cuda.max_memory_allocated()
    
    # Get memory usage after inference
    ram_after_inference = get_ram_usage()
    inference_time = time.time() - start_time
    
    # Clean up GPU memory and run garbage collection
    torch.cuda.empty_cache()
    gc.collect()
    
    # Get final peak memory usage after inference
    peak_memory_after = torch.cuda.max_memory_allocated()
    
    # Final RAM usage after inference
    ram_final = get_ram_usage()
    
    # Output results
    print(f"RAM usage after inference: {ram_after_inference:.2f} GB")
    print(f"Inference time: {inference_time:.4f} seconds")
    print(f"Peak GPU memory usage before inference: {peak_memory_before / 1024**2:.2f} MB")
    print(f"Peak GPU memory usage during inference: {peak_memory_during_inference / 1024**2:.2f} MB")
    print(f"Peak GPU memory usage after inference: {peak_memory_after / 1024**2:.2f} MB")
    print(f"Final RAM usage: {ram_final:.2f} GB")

if __name__ == "__main__":
    main()
