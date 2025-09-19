#!/usr/bin/env python3
"""
Example usage of the Oxford Loss Landscapes package.

This script demonstrates how to:
1. Create a simple neural network
2. Wrap it with the loss landscape interface
3. Compute loss landscapes in different ways
"""

#%%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
#%%
# Try to import the package
try:
    import oxford_loss_landscapes as oll
    from oxford_loss_landscapes.model_interface.model_wrapper import SimpleModelWrapper
    from oxford_loss_landscapes.hessian import get_eigenstuff, get_hessian
    from oxford_loss_landscapes.hessian.utilities import copy_wts_into_model
    from oxford_loss_landscapes.metrics import Loss
    print(f"✓ Successfully imported oxford_loss_landscapes version {oll.__version__}")
except ImportError as e:
    print(f"✗ Failed to import package: {e}")
    exit(1)

#%%
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
    """Main function to demonstrate the usage of the Oxford Loss Landscapes package.
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

    outputs = model_wrapper.forward(X)
    loss = criterion(outputs, y)
    print(f"   Initial loss: {loss.item():.4f}")

    print("4. Getting Hessian")

    hessian = get_hessian(model, loss)

    eigval, eigvec = get_eigenstuff(hessian, method="numpy")

    model_dir_1 = copy_wts_into_model(eigvec[0], model)
    model_dir_2 = copy_wts_into_model(eigvec[-1], model)


    # Create a metric to evaluate loss
    print("5. Creating loss metric...")
    try:

        loss_metric = Loss(criterion, X, y)
        
        # Test loss landscape functions
        print("5. Computing loss at current point...")
        current_loss = oll.point(model_wrapper, loss_metric)
        print(f"   Current loss: {current_loss:.4f}")
        
        # Simple visualization example
        print("6. Creating simple loss visualization...")
        try:
            # Compute a 2D loss landscape
            loss_landscape = oll.planar_interpolation(
                model_wrapper,
                model_dir_1,
                model_dir_2,
                loss_metric,
                eigen_models=True,
                steps=200
            )
            
            # Plot the line
            save_figure_name="example_hessian_basic_usage.png"
            plt.figure(figsize=(6, 5))
            plt.imshow(loss_landscape, extent=(-1, 1, -1, 1), origin='lower', aspect='auto')
            plt.xlabel('Eigenvector 1 direction')
            plt.ylabel('Eigenvector 2 direction')
            plt.title('2D Loss Landscape (Eigenvector Directions)')
            plt.colorbar(label='Loss')
            plt.savefig(save_figure_name, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   Saved loss landscape plot to '{save_figure_name}'")
        except (ValueError, RuntimeError, TypeError) as e:
            print(f"   Visualization skipped: {e}")
        
        print("\n✓ Example completed successfully!")
    except ImportError as e:
        print(f"   Could not import Loss metric: {e}")
        print("   This is expected for some package configurations.")


if __name__ == "__main__":
    main()

