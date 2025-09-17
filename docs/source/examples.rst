Examples
========

Quick Start
-----------

.. code-block:: python

   import torch
   import torch.nn as nn
   import oxford_loss_landscapes as oll

   # Create a simple model and loss function
   model = nn.Sequential(
       nn.Linear(10, 5),
       nn.ReLU(),
       nn.Linear(5, 1)
   )
   criterion = nn.MSELoss()

   # Generate some dummy data
   inputs = torch.randn(100, 10)
   targets = torch.randn(100, 1)

   # Wrap the model
   model_wrapper = oll.ModelWrapper(model, criterion, inputs, targets)

   # Compute a random 2D loss landscape
   landscape = oll.random_plane(model_wrapper, distance=1.0, steps=25)
   print(f"Loss landscape shape: {landscape.shape}")

   # Compute loss at current parameters
   loss_value = oll.point(model_wrapper)
   print(f"Current loss: {loss_value}")
