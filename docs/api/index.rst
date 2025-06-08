API Reference
=============

This section provides detailed documentation for all JaxABM classes and functions.

Core Components
---------------

.. toctree::
   :maxdepth: 2

   model
   agents
   analysis

Package Overview
----------------

.. currentmodule:: jaxabm

The main JaxABM package provides the following modules:

Core Model Classes
^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   
   Model
   AgentCollection
   ModelConfig

Analysis and Calibration
^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   
   analysis.ModelCalibrator
   analysis.SensitivityAnalysis
   analysis.EnsembleCalibrator

Utilities
^^^^^^^^^

.. autosummary::
   :toctree: generated
   
   utils.random_seed
   utils.batch_parallel
   utils.safe_divide

Legacy Support
^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   
   agentpy.Model
   agentpy.AgentList
   agentpy.Parameter

Quick Reference
---------------

Common Classes
^^^^^^^^^^^^^^

.. autoclass:: jaxabm.Model
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: jaxabm.AgentCollection
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: jaxabm.analysis.ModelCalibrator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: jaxabm.analysis.SensitivityAnalysis
   :members:
   :undoc-members:
   :show-inheritance: 