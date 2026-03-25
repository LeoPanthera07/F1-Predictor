"""
Update F1 dataset to include all historical and current races up to March 2026.
Uses the Kaggle F1 World Championship dataset.
"""

import os
import zipfile
import subprocess
import sys

def install_kaggle():
    try:
        import kaggle
    except ImportError:
        print("Installing kaggle package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])

def download_f1_data(out_dir: str = "data"):
    # Ensure kaggle is installed
    install_kaggle()
    
    print("Downloading dataset from Kaggle...")
    try:
        # We use python's subprocess to call kaggle CLI so we don't need to manually auth 
        # inside the script if the user already has ~/.kaggle/kaggle.json set up
        subprocess.check_call([
            "kaggle", "datasets", "download", 
            "-d", "rohanrao/formula-1-world-championship-1950-2020",
            "-p", out_dir
        ])
    except subprocess.CalledProcessError as e:
        print("Error downloading from Kaggle.")
        print("Ensure you have set up your Kaggle API token at ~/.kaggle/kaggle.json")
        print("You can get one from your Kaggle Account settings -> 'Create New API Token'")
        sys.exit(1)

    zip_path = os.path.join(out_dir, "formula-1-world-championship-1950-2020.zip")
    
    if os.path.exists(zip_path):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(out_dir)
        
        # Clean up the zip file
        os.remove(zip_path)
        print("Extraction complete. Raw F1 CSV files are now in your data/ folder.")
    else:
        print("Downloaded zip file not found. Something went wrong.")

if __name__ == "__main__":
    download_f1_data()
