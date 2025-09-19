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

from torchvision.models import vgg16
from torchvision import transforms
from datasets import load_dataset
#%%
# Try to import the package
try:
    import oxford_loss_landscapes as oll
    from oxford_loss_landscapes.model_interface.model_wrapper import SimpleModelWrapper
    from oxford_loss_landscapes.metrics import Loss
    print(f"✓ Successfully imported oxford_loss_landscapes version {oll.__version__}")
except ImportError as e:
    print(f"✗ Failed to import package: {e}")
    exit(1)


# %%

def main():
    """Main function to demonstrate the usage of the Oxford Loss Landscapes package with a CNN."""

    print("Oxford Loss Landscapes - Example Usage For PyTorch VGG16")
    print("=" * 40)

    print(" ---> IMPORTANT NOTE: This example requires access to the 'Maysee/tiny-imagenet' dataset.")
    print(" ---> If you do not have access to this dataset, please skip this example.")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    #%%
    # Create model and data
    print("1. Creating model and data...")
    model = vgg16(weights='IMAGENET1K_V1')
    _ = model.eval()
    #%%
    # Getting a single image from ImageNet 1k

    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    try:
        ds = load_dataset('Maysee/tiny-imagenet', split='train')
    except Exception:
        print("   Could not load 'Maysee/tiny-imagenet' dataset. Please ensure you have access to this dataset.")
        return
    #%%
    input_img = ds[0]['image'].convert('RGB')
    input_img = preprocess(input_img)
    dummy_input = input_img.unsqueeze(0)  # Add batch dimension
    output = model(dummy_input)

    pred_label = torch.argmax(output, dim=1).item()
    true_label = ds[0]['label']
    print(f"   Sample image true label: {true_label}, predicted label: {pred_label}")

    #%%
    # Wrap the model
    print("2. Wrapping model with loss landscape interface...")
    model_wrapper = SimpleModelWrapper(model)
    #%%
    # Create a metric to evaluate loss
    print("3. Creating loss metric...")
    try:

        loss_metric = Loss(nn.MSELoss(), dummy_input, output)
        
        # Test loss landscape functions
        print("4. Computing loss at current point...")
        current_loss = oll.point(model_wrapper, loss_metric)
        print(f"   Current loss: {current_loss:.4f}")

        # Simple visualization example
        print("5. Creating simple loss visualization...")
        try:
            # Compute a 1D loss line
            line_losses = oll.random_line(
                model_start=model_wrapper,
                metric=loss_metric,
                distance=1,
                steps=21,
                normalization='model'
            )

            # Plot the line
            save_figure_name = "example_cnn_usage_1D.png"
            plt.figure(figsize=(8, 5))
            plt.plot(np.linspace(-0.5, 0.5, len(line_losses)), line_losses, 'b-', linewidth=2)
            plt.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Current parameters')
            plt.xlabel('Distance from current parameters')
            plt.ylabel('Loss')
            plt.title('1D Loss Landscape (Random Direction)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(save_figure_name, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   Saved loss landscape plot to {save_figure_name}")
            
        except (ValueError, RuntimeError, TypeError, FileNotFoundError) as e:
            print(f"   Visualization skipped: {e}")

        print("6. Create 2D loss landscape visualization...")
        try:
            # Compute a 2D loss plane
            plane_losses = oll.random_plane(model_wrapper, loss_metric, distance=1.0, steps=21, normalization='model')
            
            # Plot the plane
            save_figure_name = 'example_cnn_usage_2D.png'
            X = np.linspace(-0.5, 0.5, plane_losses.shape[0])
            Y = np.linspace(-0.5, 0.5, plane_losses.shape[1])
            X, Y = np.meshgrid(X, Y)
            
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, np.log10(np.maximum(plane_losses, 1e-10)), cmap='viridis', edgecolor='none', alpha=0.8)
            ax.set_xlabel('Direction 1')
            ax.set_ylabel('Direction 2')
            ax.set_zlabel('Loss')
            ax.set_title('2D Loss Landscape (Random Plane)')
            plt.savefig(save_figure_name, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   Saved 2D loss landscape plot to {save_figure_name}")
        except (ValueError, RuntimeError, TypeError) as e:
            print(f"   2D Visualization skipped: {e}")  
        
        print("\n✓ Example completed successfully!")   
    except ImportError as e:
        print(f"   Could not import Loss metric: {e}")
        print("   This is expected for some package configurations.")


# %%
if __name__ == "__main__":
    main()
