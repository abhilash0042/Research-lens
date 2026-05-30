import os
from datasets import load_dataset

def download_sciq():
    print("Downloading SciQ dataset from HuggingFace...")
    # This will download and cache the SciQ dataset locally
    dataset = load_dataset("sciq")
    
    # Save a local copy of the dataset to the data/sciq directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sciq")
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Saving dataset to {data_dir}...")
    dataset.save_to_disk(data_dir)
    print("Download and save complete!")
    print(f"Dataset structure: {dataset}")

if __name__ == "__main__":
    download_sciq()
