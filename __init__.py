"""
JaxABM: JAX-Accelerated Agent-Based Modeling Framework

This package provides a high-performance agent-based modeling framework
with JAX acceleration for GPU-powered simulations, vectorization, and automatic differentiation.
It also includes a legacy module for traditional (non-JAX) agent-based modeling.
"""

# Standard JaxABM components
from jaxabm.legacy import Agent, AgentSet, DataCollector
from jaxabm.model import Model

# Legacy components that may not be implemented yet
# from jaxabm.legacy.space import MultiGrid, SingleGrid, ContinuousSpace, NetworkGrid, PropertyLayer
# from jaxabm.legacy.batchrunner import batch_run
# from jaxabm.legacy.visualization import JaxABMVisualization

# Try to import JAX components if installed
try:
    from jaxabm.core import ModelConfig
    from jaxabm.agent import AgentType, AgentCollection
    from jaxabm.analysis import SensitivityAnalysis, ModelCalibrator 
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

__version__ = "0.1.0"