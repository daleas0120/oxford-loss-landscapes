"""
Model downloading utilities for oxford_loss_landscapes package.

This module provides utilities for downloading models and datasets
from various sources including Zenodo.
"""

import os
from io import BytesIO
import argparse
import zipfile
import requests
from tqdm import tqdm


def download_zenodo_model(record_id, output_file, extract_dir=None):
    """
    Download a specific model file from Zenodo by record ID.
    
    Args:
        record_id (str): Zenodo record ID
        model_filename (str): Specific model filename to download
        extract_dir (str, optional): Directory to extract files to if ZIP
    """

    zip_file = download_zenodo_zip(record_id, output_file)

    if extract_dir:
        extracted_files = extract_zip(zip_file, extract_dir)
        return {'zip_file': zip_file, 'extracted_files': extracted_files}

    return {'zip_file': zip_file}


def download_zenodo_zip(record_id, output_file="models.zip"):
    """
    Download a ZIP file from Zenodo by record ID.
    
    Args:
        record_id (str): Zenodo record ID
        output_file (str): Output filename for downloaded ZIP
        extract_dir (str, optional): Directory to extract files to
    
    Returns:
        dict: Information about download and extraction
            - 'zip_file': Path to downloaded ZIP file
            - 'extracted_files': List of extracted file paths (if extracted)
    """
    url = f"https://zenodo.org/api/records/{record_id}/files"
    
    try:
        response = requests.get(url, timeout=30)  # Add 30 second timeout
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
        response = requests.get(download_url, stream=True, timeout=30)  # Add 30 second timeout
        response.raise_for_status()
        
        with open(output_file, 'wb') as f:
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=output_file) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"Downloaded: {output_file}")
        
        return output_file
        
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download from Zenodo: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Download error: {e}") from e


def extract_zip(zip_path, extract_dir):
    """
    Extract a ZIP file to a directory and return list of extracted files.
    
    Args:
        zip_path (str): Path to ZIP file
        extract_dir (str): Directory to extract to
        
    Returns:
        list: List of extracted file paths
    """
    try:
        os.makedirs(extract_dir, exist_ok=True)
        
        extracted_files = []
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files in the ZIP
            file_list = zip_ref.namelist()
            print(f"Found {len(file_list)} files in ZIP archive")
            
            # Extract all files
            zip_ref.extractall(extract_dir)
            
            # Build list of extracted file paths
            for file_name in file_list:
                extracted_path = os.path.join(extract_dir, file_name)
                if os.path.isfile(extracted_path):  # Only add actual files, not directories
                    extracted_files.append(extracted_path)
                    print(f"Extracted: {file_name}")
        
        print(f"Successfully extracted {len(extracted_files)} files to: {extract_dir}")
        return extracted_files
        
    except zipfile.BadZipFile:
        raise ValueError(f"Invalid ZIP file: {zip_path}")
    except Exception as e:
        raise RuntimeError(f"Extraction error: {e}")



def main(argv=None):
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
    
    args = parser.parse_args(argv)
    
    try:
        if args.record_id:
            result = download_zenodo_model(args.record_id, args.output, args.extract_dir)
            print("Download completed successfully!")
            print(f"ZIP file: {result['zip_file']}")
            if result['extracted_files']:
                print(f"Extracted {len(result['extracted_files'])} files")
        else:
            print("Please provide a --record-id to download from Zenodo")
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
