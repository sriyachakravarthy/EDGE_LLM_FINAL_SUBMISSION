import pandas as pd

# Data dictionary
data = {
    "Model": ["Llama3", "Qwen", "Phi"],
    "Commonsenseqa_gen": [22.11, 0, 65.68],
    "FewChid_gen": [13.59, 0.55, 12.29],
    "bbh_gen": [7.87, 1.84, 59.4],
    "HumanEval": [0.61, 0, 31.1],
    "GSM8K": [1.9, 0.83, 61.56],
    "TruthfulQA": [0.18, 0, 0.18],
    "Memory": ["8.755 GB", "7 GB", "6.143 GB"],
    "Throughput": [
        "71.61 inf/s 0.014 s",
        "77.73 inf/s 0.0129 s",
        "13.53 inf/s 0.0739 s",
    ],
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save as CSV
csv_file = "model_performance.csv"
df.to_csv(csv_file, index=False)

print(f"CSV file '{csv_file}' created successfully!")
