JaxABM Documentation
====================

**JaxABM** is a high-performance agent-based modeling framework built on JAX, designed for fast, scalable, and differentiable simulations. It provides powerful tools for parameter calibration, sensitivity analysis, and model optimization using modern machine learning techniques.

Key Features
------------

- **High Performance**: Built on JAX for GPU acceleration and JIT compilation
- **Advanced Calibration**: Multiple optimization methods including reinforcement learning
- **Sensitivity Analysis**: Comprehensive tools for parameter importance analysis
- **Differentiable**: Full compatibility with JAX's automatic differentiation
- **Scalable**: Handle large-scale agent populations efficiently
- **Flexible**: Support for custom agent behaviors and model architectures

Quick Start
-----------

Install JaxABM:

.. code-block:: bash

   pip install jaxabm

Basic usage:

.. code-block:: python

   import jaxabm as jx
   
   # Create a simple model
   model = jx.Model()
   
   # Add agents
   agents = jx.AgentCollection("traders", 1000)
   model.add_agent_collection(agents)
   
   # Run simulation
   results = model.run(steps=100)

Documentation Contents
---------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   installation
   quickstart
   tutorials/index
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: Core Features
   
   calibration/index
   sensitivity/index
   models/index
   agents/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development
   
   contributing
   changelog

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 