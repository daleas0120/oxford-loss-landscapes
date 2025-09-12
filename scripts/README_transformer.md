# Transformer Model Download Script

## Overview

The restructured `get_transformer_model.py` script provides a robust way to download transformer models for loss landscape analysis. It supports both downloading pre-trained models from Hugging Face and creating simple transformer models from scratch.

## Features

### 1. **Flexible Model Options**
- Download pre-trained transformers from Hugging Face Hub
- Create simple transformer models from scratch (no internet required)
- Automatic fallback to simple model if transformers library is unavailable

### 2. **Multiple Save Formats**
- **PyTorch format**: State dict only (`.pth` file) - ideal for loss landscape analysis
- **Transformers format**: Full model with config and tokenizer - for complete functionality
- **Both formats**: Save in both formats simultaneously

### 3. **Model Information**
- Detailed model statistics (parameters, size, architecture)
- Configuration preservation
- Comprehensive metadata saving

### 4. **Example Generation**
- Automatically creates usage examples
- Shows integration with oxford_loss_landscapes
- Demonstrates loss computation and model wrapping

## Usage

### Command Line Interface

```bash
# Basic usage - create simple transformer
python get_transformer_model.py --simple

# Download pre-trained model (requires transformers library)
python get_transformer_model.py --model "distilbert-base-uncased"

# Specify output directory and format
python get_transformer_model.py --simple --output-dir ./my_models --format pytorch

# Create example script
python get_transformer_model.py --simple --create-example
```

### Options

- `--model MODEL`: Transformer model name from Hugging Face (default: distilbert-base-uncased)
- `--output-dir DIR`: Output directory for saved models (default: ../models)
- `--format FORMAT`: Save format - pytorch, transformers, or both (default: pytorch)
- `--simple`: Create simple transformer from scratch instead of downloading
- `--create-example`: Generate example usage script

## Simple Transformer Specifications

When using `--simple`, the script creates a transformer with:

- **Hidden size**: 256
- **Number of layers**: 4
- **Attention heads**: 8
- **Vocabulary size**: 10,000
- **Total parameters**: ~8.5M
- **Model size**: ~32.6 MB

## Output Files

### For Simple Transformer (`--simple`)
- `simple_transformer_weights.pth`: PyTorch state dict with metadata
- `load_transformer_example.py`: Example usage script (if `--create-example`)

### For Downloaded Models
- `transformer_weights.pth`: PyTorch state dict (if pytorch format)
- `transformer_model/`: Full model directory (if transformers format)
- `load_transformer_example.py`: Example usage script (if `--create-example`)

## Integration with Oxford Loss Landscapes

The generated models are designed for seamless integration:

```python
# Load the model
from load_transformer_example import create_model_for_analysis
model = create_model_for_analysis()

# Use with loss landscapes
from src.oxford_loss_landscapes.model_interface.model_wrapper import SimpleModelWrapper
from src.oxford_loss_landscapes.main import point

model_wrapper = SimpleModelWrapper(model)
# ... define metric and use with loss landscape functions
```

## Error Handling

The script includes robust error handling:

1. **Missing transformers library**: Automatically falls back to simple model creation
2. **Network issues**: Graceful degradation with informative messages
3. **Invalid model names**: Clear error messages and alternatives
4. **Directory creation**: Automatic creation of output directories

## Dependencies

### Required
- `torch`: PyTorch framework
- `pathlib`: Path handling (built-in)
- `argparse`: Command line parsing (built-in)

### Optional
- `transformers`: For downloading pre-trained models from Hugging Face

## Example Workflow

1. **Download/Create Model**:
   ```bash
   python get_transformer_model.py --simple --create-example
   ```

2. **Load in Python**:
   ```python
   from load_transformer_example import create_model_for_analysis
   model = create_model_for_analysis()
   ```

3. **Use with Loss Landscapes**:
   ```python
   from src.oxford_loss_landscapes.model_interface.model_wrapper import SimpleModelWrapper
   model_wrapper = SimpleModelWrapper(model)
   # Ready for loss landscape analysis!
   ```

## Improvements from Original

### Original Script Issues:
- ❌ No error handling
- ❌ Fixed model only
- ❌ No usage examples
- ❌ Incomplete saving format
- ❌ No documentation

### New Script Features:
- ✅ Comprehensive error handling and fallback options
- ✅ Multiple model options (simple vs. pre-trained)
- ✅ Automatic example generation
- ✅ Multiple save formats with metadata
- ✅ Command line interface with help
- ✅ Full documentation and examples
- ✅ Integration testing with loss landscapes package

## Future Enhancements

Potential future improvements:
- Support for more model architectures (GPT, T5, etc.)
- Custom model size configurations
- Batch processing for multiple models
- Integration with model compression techniques
- Support for different tokenizers and vocabularies
