Features
========

Core Loss Landscape Functions
-----------------------------

- ``point()``: Evaluate loss at current parameters
- ``linear_interpolation()``: Loss along a line between two points
- ``random_line()``: Loss along a random direction
- ``planar_interpolation()``: Loss over a 2D plane between three points  
- ``random_plane()``: Loss over a random 2D plane

Model Interface
---------------

- ``ModelWrapper``: Interface for PyTorch models
- ``SimpleModelWrapper``: Interface for simple models

Utilities
---------

- Model downloading from Zenodo datasets
- Hessian computation tools
- Dashboard data parsing utilities
