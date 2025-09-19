Installation
============

Simple Installation (Recommended)
---------------------------------

The package automatically installs all dependencies including PyTorch:

.. code-block:: bash

   pip install -e .

This single command installs:

- PyTorch (with automatic version selection for your Python version)  
- All numerical computation dependencies (NumPy, SciPy, pandas)
- Visualization tools (matplotlib, seaborn)
- Package-specific dependencies (torchdiffeq, etc.)

Development Installation
------------------------

For development with testing and documentation tools:

.. code-block:: bash

   pip install -e .[dev]

Optional Dependencies
---------------------

For transformer model support:

.. code-block:: bash

   pip install -e ".[transformers]"

For advanced visualization and analysis tools:

.. code-block:: bash

   pip install -e ".[advanced]"

This includes: plotly, ipywidgets, jupyter, scikit-learn, torchvision

Installation with requirements.txt
----------------------------------

.. code-block:: bash

   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .

Python Version Compatibility
----------------------------

- **Python 3.8–3.12**: Uses PyTorch >=1.8.0 and NumPy <2.0
- **Python 3.13+**: Uses PyTorch >=2.5.0 and NumPy >=2.0 (automatic version selection)
