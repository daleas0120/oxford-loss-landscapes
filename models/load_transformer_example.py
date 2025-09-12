#!/usr/bin/env python3
"""
Example: Loading and Using Downloaded Transformer Model

This script demonstrates how to load the downloaded transformer model
for use with the oxford_loss_landscapes package.
"""

import torch
import torch.nn as nn
from pathlib import Path

def load_transformer_weights(weights_path="../models/simple_transformer_weights.pth"):
    """Load the downloaded transformer weights."""
    
    # Load the saved weights
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    print("Model Information:")
    print(f"  Name: {checkpoint['model_name']}")
    print(f"  Architecture: {checkpoint['architecture']}")
    print(f"  Parameters: {checkpoint['num_parameters']:,}")
    
    return checkpoint

def create_model_for_analysis():
    """
    Create the transformer model for loss landscape analysis.
    """
    checkpoint = load_transformer_weights()
    
    if checkpoint.get('model_config', {}).get('type') == 'simple_transformer':
        # Create simple transformer
        print("Creating simple transformer model...")
        
        class SimpleTransformerModel(nn.Module):
            def __init__(self, vocab_size, hidden_size, num_layers, num_heads):
                super().__init__()
                self.hidden_size = hidden_size
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.pos_encoding = nn.Parameter(torch.randn(1000, hidden_size))
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size, 
                    nhead=num_heads,
                    dim_feedforward=hidden_size * 4,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.output_projection = nn.Linear(hidden_size, vocab_size)
                self.dropout = nn.Dropout(0.1)
                
            def forward(self, x):
                seq_len = x.size(1)
                embedded = self.embedding(x) + self.pos_encoding[:seq_len]
                embedded = self.dropout(embedded)
                output = self.transformer(embedded)
                logits = self.output_projection(output)
                return logits
        
        config = checkpoint['model_config']
        model = SimpleTransformerModel(
            vocab_size=config['vocab_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads']
        )
        
        # Load the weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
        
    else:
        # For downloaded models, try to use transformers library
        try:
            from transformers import AutoModel
            
            # Load the complete model
            model = AutoModel.from_pretrained("../models/transformer_model")
            return model
            
        except ImportError:
            print("transformers library not available.")
            print("For downloaded models, please install: pip install transformers")
            return None

def example_loss_landscape_analysis():
    """Example of using the model with oxford_loss_landscapes."""
    
    # Load the model
    model = create_model_for_analysis()
    if model is None:
        return
    
    print("\nExample: Setting up for loss landscape analysis...")
    
    # Create some dummy data
    batch_size, seq_len = 8, 50
    vocab_size = 10000
    
    # Random input tokens
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Define a simple loss function
    def loss_fn(model_output):
        # Simple mean squared error for demonstration
        target = torch.randn_like(model_output)
        return torch.nn.functional.mse_loss(model_output, target)
    
    # Example of using with oxford_loss_landscapes
    try:
        import sys
        sys.path.append('..')  # Add parent directory to path
        from src.oxford_loss_landscapes.model_interface.model_wrapper import ModelWrapper
        from src.oxford_loss_landscapes.main import point
        from src.oxford_loss_landscapes.metrics.sl_metrics import Loss
        
        # Create a simple metric
        from src.oxford_loss_landscapes.metrics.metric import Metric
        
        class TransformerLoss(Metric):
            def __init__(self, inputs, loss_fn):
                super().__init__()
                self.inputs = inputs
                self.loss_fn = loss_fn
                
            def __call__(self, model_wrapper):
                output = model_wrapper.forward(self.inputs)
                return self.loss_fn(output).item()
        
        # Wrap the model
        from src.oxford_loss_landscapes.model_interface.model_wrapper import SimpleModelWrapper
        model_wrapper = SimpleModelWrapper(model)
        
        # Create the metric
        metric = TransformerLoss(input_ids, loss_fn)
        
        # Compute loss at current point
        loss_value = point(model_wrapper, metric)
        print(f"Current loss value: {loss_value}")
        
        print("\n✅ Model is ready for loss landscape analysis!")
        print("You can now use this model with other oxford_loss_landscapes functions.")
        
    except ImportError as e:
        print(f"\nNote: oxford_loss_landscapes not found: {e}")
        print("The model is ready, but you'll need to set up the loss landscapes package.")

if __name__ == "__main__":
    # Example usage
    print("Loading transformer model weights...")
    checkpoint = load_transformer_weights()
    
    print("\nCreating model for analysis...")
    model = create_model_for_analysis()
    
    if model is not None:
        print("\nRunning example loss landscape analysis...")
        example_loss_landscape_analysis()
