Installation
============

Requirements
------------

JaxABM requires Python 3.8 or later and has the following dependencies:

Core Dependencies
^^^^^^^^^^^^^^^^^

- **JAX**: For high-performance computing and automatic differentiation
- **NumPy**: For numerical operations
- **SciPy**: For scientific computing utilities

Optional Dependencies
^^^^^^^^^^^^^^^^^^^^^

- **Matplotlib**: For plotting and visualization
- **Seaborn**: For advanced statistical plots
- **Pandas**: For data manipulation and analysis
- **NetworkX**: For network-based models

Installation Methods
-------------------

From PyPI (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install jaxabm

For GPU support with CUDA:

.. code-block:: bash

   # Install JAX with CUDA support first
   pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   
   # Then install JaxABM
   pip install jaxabm

From Source
^^^^^^^^^^^

For development or the latest features:

.. code-block:: bash

   git clone https://github.com/username/jaxabm.git
   cd jaxabm
   pip install -e .

Development Installation
^^^^^^^^^^^^^^^^^^^^^^^^

For contributing to JaxABM:

.. code-block:: bash

   git clone https://github.com/username/jaxabm.git
   cd jaxabm
   pip install -e ".[dev]"

This installs additional development dependencies for testing and documentation.

Verification
------------

Test your installation:

.. code-block:: python

   import jaxabm as jx
   import jax.numpy as jnp
   
   # Check JAX is working
   print(f"JAX version: {jx.__version__}")
   print(f"JAX devices: {jx.devices()}")
   
   # Simple test
   model = jx.Model()
   print("✅ JaxABM installed successfully!")

GPU Support
-----------

To use GPU acceleration, ensure you have:

1. **NVIDIA GPU** with CUDA support
2. **CUDA toolkit** installed
3. **JAX with CUDA** installed

Check GPU availability:

.. code-block:: python

   import jax
   
   print(f"JAX devices: {jax.devices()}")
   print(f"GPU available: {len(jax.devices('gpu')) > 0}")

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

**JAX Installation Issues**
   If you encounter issues with JAX installation, refer to the `JAX installation guide <https://github.com/google/jax#installation>`_.

**Import Errors**
   If you get import errors, ensure all dependencies are installed:
   
   .. code-block:: bash
   
      pip install --upgrade jaxabm

**GPU Not Detected**
   For GPU issues, check your CUDA installation and JAX GPU setup:
   
   .. code-block:: bash
   
      python -c "import jax; print(jax.devices())"

Performance Tips
----------------

For optimal performance:

1. **Use JAX transformations**: ``jit``, ``vmap``, ``pmap``
2. **Batch operations**: Process multiple agents simultaneously
3. **GPU acceleration**: Use CUDA for large-scale simulations
4. **Memory management**: Use JAX's memory-efficient operations

Next Steps
----------

- Read the :doc:`quickstart` guide
- Explore :doc:`tutorials/index`
- Check out :doc:`examples/index` 