"""Test ways to download models."""

import os
import shutil
import pytest

def test_download_zenodo_zip():
    """Test downloading a ZIP file from Zenodo and extracting it."""
    try:
        from oxford_loss_landscapes.download_models import download_zenodo_zip
    except ImportError:
        pytest.skip("Package not properly installed; skipping related tests.")
    os.makedirs("test_downloads", exist_ok=True)
    # Test downloading a small file from Zenodo
    model_path = download_zenodo_zip("12763938", output_file="test_downloads/paper-data-redundancy-v1.0.zip")
    assert os.path.exists(model_path), "Model file was not downloaded"
    # Clean up
    os.remove(model_path)
    os.rmdir(os.path.dirname(model_path))

    assert not os.path.exists(model_path), "Model file was not removed"
    assert not os.path.exists(os.path.dirname(model_path)), "Download directory was not removed"

def test_download_zenodo_model():
    """Test downloading and extracting a model from Zenodo."""
    try:
        from oxford_loss_landscapes.download_models import download_zenodo_model
    except ImportError:
        pytest.skip("Package not properly installed; skipping related tests.")

    os.makedirs("test_downloads", exist_ok=True)
    # Test downloading and extracting a small file from Zenodo
    result = download_zenodo_model("12763938", 'test_downloads/model.zip', extract_dir="test_downloads")
    assert 'zip_file' in result, "Result should contain 'zip_file'"
    assert 'extracted_files' in result, "Result should contain 'extracted_files'"

    assert not os.path.exists(os.path.join("test_downloads", result['zip_file'])), "ZIP file was not removed after extraction"
    
    for file_path in result['extracted_files']:
        assert os.path.exists(file_path), f"Extracted file {file_path} does not exist"
    
    # Clean up
    shutil.rmtree("test_downloads")

    assert not os.path.exists(os.path.join("test_downloads", result['zip_file'])), "ZIP file was not removed after extraction"
    assert not os.path.exists(os.path.dirname(os.path.join("test_downloads", result['zip_file']))), "Download directory was removed"


def test_main_download_models():
    """Test the command line interface for downloading models."""
    try:
        from oxford_loss_landscapes.download_models import main
    except ImportError:
        pytest.skip("Package not properly installed; skipping related tests.")
    
    # Prepare test arguments
    test_args = [
        "--record-id", "12763938",
        "--output", "test_downloads/paper-data-redundancy-v1.0.zip",
        "--extract-dir", "test_downloads"
    ]
    
    os.makedirs("test_downloads", exist_ok=True)
    
    try:
        exit_code = main(test_args)
        assert exit_code == 0, f"Main function exited with code {exit_code}"
        
        # Check that files were downloaded and extracted
        assert len(os.listdir("test_downloads")) > 0, "No files were extracted"
        
    finally:
        # Clean up
        if os.path.exists("test_downloads"):
            shutil.rmtree("test_downloads")

    assert not os.path.exists("test_downloads"), "Download directory was not removed"
