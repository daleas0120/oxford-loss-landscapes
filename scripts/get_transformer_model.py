#!/usr/bin/env python3
"""
Download Simple Transformer Model Script

This script downloads a simple pre-trained transformer model and saves the weights
for use with the oxford_loss_landscapes package. The script downloads a lightweight
transformer model suitable for loss landscape analysis.
"""

import os
import sys
import torch
import torch.nn as nn
import argparse
from pathlib import Path

def create_simple_transformer_model(hidden_size=256, num_layers=4, num_heads=8, vocab_size=10000):
    """
    Create a simple transformer model from scratch for loss landscape analysis.
    This doesn't require the transformers library.
    """
    
    class SimpleTransformerModel(nn.Module):
        def __init__(self, vocab_size, hidden_size, num_layers, num_heads):
            super().__init__()
            self.hidden_size = hidden_size
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.pos_encoding = nn.Parameter(torch.randn(1000, hidden_size))
            
            # Transformer encoder layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size, 
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # Output projection
            self.output_projection = nn.Linear(hidden_size, vocab_size)
            self.dropout = nn.Dropout(0.1)
            
        def forward(self, x):
            seq_len = x.size(1)
            
            # Embeddings + positional encoding
            embedded = self.embedding(x) + self.pos_encoding[:seq_len]
            embedded = self.dropout(embedded)
            
            # Transformer
            output = self.transformer(embedded)
            
            # Output projection
            logits = self.output_projection(output)
            
            return logits
    
    return SimpleTransformerModel(vocab_size, hidden_size, num_layers, num_heads)

def download_transformer_model(model_name="distilbert-base-uncased", output_dir="../models", save_format="pytorch", use_simple=False):
    """
    Download a transformer model and save the weights.
    
    Args:
        model_name (str): Name of the transformer model to download (ignored if use_simple=True)
        output_dir (str): Directory to save the model
        save_format (str): Format to save the model ('pytorch', 'transformers', or 'both')
        use_simple (bool): If True, create a simple transformer instead of downloading
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if use_simple:
        print("Creating simple transformer model from scratch...")
        
        # Create a simple transformer model
        model = create_simple_transformer_model()
        
        # Initialize with reasonable weights
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0, std=0.1)
        
        model.apply(init_weights)
        model.eval()
        
        # Save the model
        pytorch_path = output_path / "simple_transformer_weights.pth"
        print(f"Saving simple transformer weights to {pytorch_path}")
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'hidden_size': 256,
                'num_layers': 4,
                'num_heads': 8,
                'vocab_size': 10000,
                'type': 'simple_transformer'
            },
            'model_name': 'simple_transformer',
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'architecture': model.__class__.__name__,
        }, pytorch_path)
        
        # Print model information
        print("\n" + "="*50)
        print("SIMPLE TRANSFORMER MODEL CREATED")
        print("="*50)
        print(f"Model: Simple Transformer (created from scratch)")
        print(f"Architecture: {model.__class__.__name__}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Hidden size: 256")
        print(f"Number of layers: 4")
        print(f"Number of attention heads: 8")
        print(f"Vocabulary size: 10,000")
        print(f"Model size: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024:.2f} MB")
        print(f"PyTorch weights saved to: {pytorch_path}")
        print("="*50)
        
        return model, None, None
    
    else:
        # Try to use transformers library
        try:
            from transformers import AutoModel, AutoTokenizer, AutoConfig
            print(f"Downloading transformer model: {model_name}")
            
            # Download model, tokenizer, and config
            print("Loading model...")
            model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
            
            print("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            print("Loading config...")
            config = AutoConfig.from_pretrained(model_name)
            
            # Set model to evaluation mode
            model.eval()
            
            # Save in different formats based on user preference
            if save_format in ["transformers", "both"]:
                # Save in transformers format (full model with config and tokenizer)
                transformers_dir = output_path / "transformer_model"
                transformers_dir.mkdir(exist_ok=True)
                
                print(f"Saving model in transformers format to {transformers_dir}")
                model.save_pretrained(transformers_dir)
                tokenizer.save_pretrained(transformers_dir)
                
            if save_format in ["pytorch", "both"]:
                # Save just the PyTorch state dict for loss landscape analysis
                pytorch_path = output_path / "transformer_weights.pth"
                
                print(f"Saving PyTorch weights to {pytorch_path}")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_config': config.to_dict(),
                    'model_name': model_name,
                    'num_parameters': sum(p.numel() for p in model.parameters()),
                    'architecture': model.__class__.__name__,
                }, pytorch_path)
                
            # Print model information
            print("\n" + "="*50)
            print("MODEL DOWNLOAD SUMMARY")
            print("="*50)
            print(f"Model: {model_name}")
            print(f"Architecture: {model.__class__.__name__}")
            print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
            print(f"Model size: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024:.2f} MB")
            
            if save_format in ["transformers", "both"]:
                print(f"Transformers format saved to: {transformers_dir}")
            if save_format in ["pytorch", "both"]:
                print(f"PyTorch weights saved to: {pytorch_path}")
            print("="*50)
            
            return model, tokenizer, config
            
        except ImportError:
            print("transformers library not found.")
            print("Creating simple transformer model instead...")
            print("(To use pre-trained models, install transformers: pip install transformers)")
            return download_transformer_model(model_name, output_dir, save_format, use_simple=True)
            
        except Exception as e:
            print(f"Error downloading model: {e}")
            print("Creating simple transformer model instead...")
            return download_transformer_model(model_name, output_dir, save_format, use_simple=True)


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Download a simple transformer model for loss landscape analysis")
    
    parser.add_argument("--model", default="distilbert-base-uncased", 
                       help="Transformer model name (default: distilbert-base-uncased)")
    parser.add_argument("--output-dir", default="../models",
                       help="Output directory for saved models (default: ../models)")
    parser.add_argument("--format", choices=["pytorch", "transformers", "both"], default="pytorch",
                       help="Save format: pytorch (weights only), transformers (full), or both (default: pytorch)")
    parser.add_argument("--simple", action="store_true",
                       help="Create a simple transformer from scratch instead of downloading")
    
    args = parser.parse_args()
    
    print("Oxford Loss Landscapes - Transformer Model Downloader")
    print("="*55)
    
    # Download the model
    model, tokenizer, config = download_transformer_model(
        model_name=args.model,
        output_dir=args.output_dir,
        save_format=args.format,
        use_simple=args.simple
    )
    
    print("\n✅ Transformer model download completed successfully!")

if __name__ == "__main__":
    main()