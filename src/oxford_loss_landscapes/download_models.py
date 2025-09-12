"""
Model downloading utilities for oxford_loss_landscapes package.

This module provides utilities for downloading models and datasets
from various sources including Zenodo.
"""

import os
import requests
import zipfile
from io import BytesIO
from tqdm import tqdm
import argparse


def download_zenodo_zip(record_id, output_file="models.zip", extract_dir=None):
    """
    Download a ZIP file from Zenodo by record ID.
    
    Args:
        record_id (str): Zenodo record ID
        output_file (str): Output filename for downloaded ZIP
        extract_dir (str, optional): Directory to extract files to
    
    Returns:
        str: Path to downloaded file
    """
    url = f"https://zenodo.org/api/records/{record_id}/files"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        files = response.json()['entries']
        
        if not files:
            raise ValueError(f"No files found for Zenodo record {record_id}")
        
        # Get the first (usually only) file
        file_info = files[0]
        download_url = file_info['links']['content']
        file_size = file_info['size']
        
        print(f"Downloading {file_info['key']} ({file_size} bytes)...")
        
        # Download with progress bar
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        with open(output_file, 'wb') as f:
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=output_file) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"Downloaded: {output_file}")
        
        # Extract if requested
        if extract_dir:
            extract_zip(output_file, extract_dir)
        
        return output_file
        
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download from Zenodo: {e}")
    except Exception as e:
        raise RuntimeError(f"Download error: {e}")


def extract_zip(zip_path, extract_dir):
    """
    Extract a ZIP file to a directory.
    
    Args:
        zip_path (str): Path to ZIP file
        extract_dir (str): Directory to extract to
    """
    try:
        os.makedirs(extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        print(f"Extracted to: {extract_dir}")
        
    except zipfile.BadZipFile:
        raise ValueError(f"Invalid ZIP file: {zip_path}")
    except Exception as e:
        raise RuntimeError(f"Extraction error: {e}")


def download_model(model_name="default", output_dir="./models"):
    """
    Download a specific model by name.
    
    Args:
        model_name (str): Name of model to download
        output_dir (str): Directory to save model to
        
    Returns:
        str: Path to downloaded model
    """
    # Placeholder for model-specific download logic
    models = {
        "default": "1234567",  # Example Zenodo record ID
    }
    
    if model_name not in models:
        available = ", ".join(models.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    
    record_id = models[model_name]
    output_file = os.path.join(output_dir, f"{model_name}_model.zip")
    
    os.makedirs(output_dir, exist_ok=True)
    
    return download_zenodo_zip(record_id, output_file, output_dir)


def main():
    """Command line interface for downloading models."""
    parser = argparse.ArgumentParser(description="Download models from Zenodo")
    parser.add_argument(
        "--output", 
        default="models.zip",
        help="Output filename for downloaded ZIP file"
    )
    parser.add_argument(
        "--extract-dir",
        help="Directory to extract data to"
    )
    parser.add_argument(
        "--record-id",
        help="Zenodo record ID to download from"
    )
    
    args = parser.parse_args()
    
    try:
        if args.record_id:
            download_zenodo_zip(args.record_id, args.output, args.extract_dir)
        else:
            print("Please provide a --record-id to download from Zenodo")
            print("Example: oxford-download-models --record-id 1234567 --extract-dir ./models")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
