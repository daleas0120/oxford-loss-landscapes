#!/usr/bin/env python3
"""
Example usage of the Oxford Loss Landscapes package.

This script demonstrates how to:
1. Create a simple neural network
2. Wrap it with the loss landscape interface
3. Compute loss landscapes in different ways
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM

# Try to import the package
try:
    import oxford_loss_landscapes as oll
    from oxford_loss_landscapes.model_interface.model_wrapper import TransformerModelWrapper
    print(f"✓ Successfully imported oxford_loss_landscapes version {oll.__version__}")
except ImportError as e:
    print(f"✗ Failed to import package: {e}")
    exit(1)


def main():
    print("Oxford Loss Landscapes - Example Usage For Transformers")
    print("=" * 40)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create model and data
    print("1. Creating model and data...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained('gpt2')

    text = "Look for the bear necessities, the simple bear necessities."
    print(f"    Sample text: {text}")
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)

    criterion = nn.CrossEntropyLoss()
    
    # Wrap the model
    print("2. Wrapping model with loss landscape interface...")
    model_wrapper = TransformerModelWrapper(model, tokenizer)
    
    # Create a metric to evaluate loss
    print("3. Creating loss metric...")
    try:
        from oxford_loss_landscapes.metrics import LanguageModelingLoss
        loss_metric = LanguageModelingLoss(encoded_input['input_ids'])
        
        # Test loss landscape functions
        print("4. Computing loss at current point...")
        current_loss = oll.point(model_wrapper, loss_metric)
        print(f"   Current loss: {current_loss:.4f}")
        
        # Simple visualization example
        print("5. Creating simple loss visualization...")
        try:
            # Compute a 1D loss line
            model_path = 'transformer_loss_landscape_example.png'
            line_losses = oll.random_line(model_wrapper, loss_metric, distance=1.0, steps=25)
            
            # Plot the line
            plt.figure(figsize=(8, 5))
            plt.plot(np.linspace(-1, 1, len(line_losses)), line_losses, 'b-', linewidth=2)
            plt.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Current parameters')
            plt.xlabel('Distance from current parameters')
            plt.ylabel('Loss')
            plt.title('1D Loss Landscape (Random Direction)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(model_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   Saved loss landscape plot to {model_path}")
        except Exception as e:
            print(f"   Visualization skipped: {e}")
        
        print("\n✓ Example completed successfully!")
    except ImportError as e:
        print(f"   Could not import Loss metric: {e}")
        print("   This is expected for some package configurations.")
    
    print("\nAvailable functions in oxford_loss_landscapes:")
    functions = [attr for attr in dir(oll) if not attr.startswith('_') and callable(getattr(oll, attr))]
    for func in sorted(functions)[:10]:  # Show first 10
        print(f"   - {func}")
    if len(functions) > 10:
        print(f"   ... and {len(functions) - 10} more")


if __name__ == "__main__":
    main()
