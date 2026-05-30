import os
from datasets import load_dataset

def download_qasper():
    print("Downloading QASPER dataset from HuggingFace...")
    # This will download and cache the dataset locally
    dataset = load_dataset("allenai/qasper", trust_remote_code=True)
    
    # Save a local copy of the dataset metadata or structures if needed
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "qasper")
    os.makedirs(data_dir, exist_ok=True)
    
    # We can save it to disk for offline use
    print(f"Saving dataset to {data_dir}...")
    dataset.save_to_disk(data_dir)
    print("Download and save complete!")
    print(f"Dataset structure: {dataset}")

if __name__ == "__main__":
    download_qasper()
