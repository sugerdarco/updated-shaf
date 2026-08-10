import os
from datasets import load_dataset
import json

def download_and_save_sample(dataset_name, config_name, split, output_file, num_samples=100):
    print(f"Downloading {dataset_name} ({config_name}) - {split} split...")
    try:
        # Load dataset
        if config_name:
            dataset = load_dataset(dataset_name, config_name, split=split, streaming=True, trust_remote_code=True)
        else:
            dataset = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True)
        
        # Take a sample
        sample = dataset.take(num_samples)
        
        # Save to JSONL
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in sample:
                f.write(json.dumps(item) + "\n")
        print(f"Successfully saved {num_samples} samples to {output_file}")
    except Exception as e:
        print(f"Failed to download {dataset_name}: {str(e)}")

def main():
    base_dir = "/home/r4xok/Post_Grade_Project/TEMP/updated-shaf/dataset"
    os.makedirs(base_dir, exist_ok=True)
    
    datasets_to_fetch = [
        {"name": "cais/mmlu", "config": "all", "split": "test", "file": "mmlu_sample.jsonl"},
        {"name": "allenai/ai2_arc", "config": "ARC-Challenge", "split": "test", "file": "arc_c_sample.jsonl"},
        {"name": "openai/gsm8k", "config": "main", "split": "test", "file": "gsm8k_sample.jsonl"},
        {"name": "piqa", "config": None, "split": "test", "file": "piqa_sample.jsonl"},
        {"name": "mandarjoshi/trivia_qa", "config": "rc", "split": "validation", "file": "triviaqa_sample.jsonl"},
        # Note: NQ can be extremely heavy. We will try the default QA variant.
        {"name": "google-research-datasets/natural_questions", "config": "default", "split": "validation", "file": "nq_sample.jsonl"}
    ]
    
    for ds in datasets_to_fetch:
        output_path = os.path.join(base_dir, ds["file"])
        if not os.path.exists(output_path):
            download_and_save_sample(ds["name"], ds["config"], ds["split"], output_path)
        else:
            print(f"Skipping {ds['name']}, file already exists.")

if __name__ == "__main__":
    main()
