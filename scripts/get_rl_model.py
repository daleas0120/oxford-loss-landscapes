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

def create_simple_rl_model():
    """
    Create a simple RL model from scratch for loss landscape analysis.
    This doesn't require the transformers library.
    """
    
    class SimpleRLModel(nn.Module):
        def __init__(self, input_size=4, hidden_size=128, num_layers=2, output_size=2):
            super().__init__()
            self.hidden_size = hidden_size
            
            # Simple feedforward layers
            layers = []
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_size, output_size))
            
            self.network = nn.Sequential(*layers)
            
        def forward(self, x):
            return self.network(x)
    
    return SimpleRLModel()

def main():
    """Main function to create and save the RL model."""
    parser = argparse.ArgumentParser(description="Create and save a simple RL model.")
    parser.add_argument(
        "--save_path",
        type=str,
        default="../models/simple_rl_model_weights.pth",
        help="Path to save the model weights.",
    )
    args = parser.parse_args()
    
    # Create model
    model = create_simple_rl_model()
    
    # Save the model weights
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_name': 'simple_rl_model',
        'architecture': 'Feedforward Neural Network',
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'model_state_dict': model.state_dict(),
    }, save_path)
    
    print(f"Model weights saved to {save_path}")

if __name__ == "__main__":
    # Example usage
    print("Creating and saving a simple RL model...")
    main()