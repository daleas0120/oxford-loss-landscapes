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

# Try to import the package
try:
    import oxford_loss_landscapes as oll
    from oxford_loss_landscapes.model_interface.model_wrapper import SimpleModelWrapper
    print(f"✓ Successfully imported oxford_loss_landscapes version {oll.__version__}")
except ImportError as e:
    print(f"✗ Failed to import package: {e}")
    exit(1)


def create_simple_model():
    """Create a simple neural network for demonstration."""
    model = nn.Sequential(
        nn.Linear(2, 10),
        nn.ReLU(),
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )
    return model


def generate_data(n_samples=100):
    """Generate simple synthetic data for demonstration."""
    # Generate 2D input data
    X = torch.randn(n_samples, 2)
    # Simple target: sum of squares
    y = (X[:, 0]**2 + X[:, 1]**2).unsqueeze(1) + 0.1 * torch.randn(n_samples, 1)
    return X, y


def main():
    """
    Main function to demonstrate the usage of the Oxford Loss Landscapes package.
    """
    print("Oxford Loss Landscapes - Example Usage")
    print("=" * 40)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create model and data
    print("1. Creating model and data...")
    model = create_simple_model()
    X, y = generate_data(100)
    criterion = nn.MSELoss()
    
    # Wrap the model
    print("2. Wrapping model with loss landscape interface...")
    model_wrapper = SimpleModelWrapper(model)
    
    # Test forward pass
    print("3. Testing forward pass...")
    with torch.no_grad():
        outputs = model_wrapper.forward(X)
        loss = criterion(outputs, y)
        print(f"   Initial loss: {loss.item():.4f}")
    
    # Create a metric to evaluate loss
    print("4. Creating loss metric...")
    try:
        from oxford_loss_landscapes.metrics import Loss
        loss_metric = Loss(criterion, X, y)
        
        # Test loss landscape functions
        print("5. Computing loss at current point...")
        current_loss = oll.point(model_wrapper, loss_metric)
        print(f"   Current loss: {current_loss:.4f}")
        
        # Simple visualization example
        print("6. Creating simple loss visualization...")
        try:
            # Compute a 1D loss line
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
            plt.savefig('loss_landscape_example.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   Saved loss landscape plot to 'loss_landscape_example.png'")
        except (ValueError, RuntimeError, TypeError, plt.matplotlib.MatplotlibError) as e:
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
