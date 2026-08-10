import os
import json
from datasets import load_dataset

def download_full_dataset(dataset_name, config_name, splits, output_dir):
    print(f"\n==========================================")
    print(f"Downloading full dataset: {dataset_name} (config: {config_name})")
    print(f"==========================================")
    try:
        os.makedirs(output_dir, exist_ok=True)
        if config_name:
            ds = load_dataset(dataset_name, config_name)
        else:
            ds = load_dataset(dataset_name)
        
        print(f"Successfully loaded dataset structure for {dataset_name}:")
        for split_name in ds.keys():
            num_rows = len(ds[split_name])
            print(f"  - Split '{split_name}': {num_rows} examples")
            out_file = os.path.join(output_dir, f"{dataset_name.replace('/', '_')}_{config_name or 'default'}_{split_name}.jsonl")
            print(f"    Exporting to {out_file}...")
            ds[split_name].to_json(out_file)
            print(f"    Saved {split_name} ({os.path.getsize(out_file) / (1024**2):.2f} MB)")
            
    except Exception as e:
        print(f"Failed to download {dataset_name}: {str(e)}")

def main():
    base_dir = "/home/r4xok/Post_Grade_Project/TEMP/updated-shaf/dataset/full_datasets"
    os.makedirs(base_dir, exist_ok=True)
    
    datasets_to_fetch = [
        {"name": "cais/mmlu", "config": "all", "splits": ["test", "validation", "dev"]},
        {"name": "allenai/ai2_arc", "config": "ARC-Challenge", "splits": ["train", "test", "validation"]},
        {"name": "openai/gsm8k", "config": "main", "splits": ["train", "test"]},
        {"name": "ybisk/piqa", "config": None, "splits": ["train", "validation", "test"]},
        {"name": "mandarjoshi/trivia_qa", "config": "rc", "splits": ["validation", "test"]},
        {"name": "google-research-datasets/natural_questions", "config": "default", "splits": ["validation"]}
    ]
    
    for ds in datasets_to_fetch:
        download_full_dataset(ds["name"], ds["config"], ds["splits"], base_dir)

if __name__ == "__main__":
    main()
