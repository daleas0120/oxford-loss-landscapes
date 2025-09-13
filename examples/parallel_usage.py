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
import time

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
        nn.Linear(100, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )
    return model


def generate_data(n_samples=100):
    """Generate simple synthetic data for demonstration."""
    # Generate 2D input data
    X = torch.randn(n_samples, 2)
    # Simple target: sum of squares
    y = (X[:, 0]**2 + X[:, 1]**2).unsqueeze(1) + 0.1 * torch.randn(n_samples, 1)
    return X, y


def generate_heavier_data(n_samples=50000, n_features=100):
    """Generate larger synthetic dataset."""
    X = torch.randn(n_samples, n_features)
    y = (X[:, 0]**2 + X[:, 1]**2).unsqueeze(1) + 0.1 * torch.randn(n_samples, 1)
    return X, y


def main():
    start_time = time.perf_counter()
    print("Oxford Loss Landscapes - Example Usage")
    print("=" * 40)
    
    # Set random seed for reproducibility
    torch.manual_seed(17)
    
    # Create model and data
    print("1. Creating model and data...")
    model = create_simple_model()
    X, y = generate_heavier_data()
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
        
        line_losses = oll.random_line(model_wrapper, loss_metric, distance=1.0, steps=100)
            
        print("\n✓ Example completed successfully!")
    except ImportError as e:
        print(f"   Could not import Loss metric: {e}")
        print("   This is expected for some package configurations.")


    end_time = time.perf_counter()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
