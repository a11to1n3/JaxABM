"""
JaxABM: An agent-based modeling framework accelerated by JAX.

This package provides tools for building, running, and analyzing agent-based models
with traditional Python interfaces or with JAX acceleration.

The legacy version (non-JAX) is available directly from the package root.
The JAX-accelerated version is available when JAX is installed.
"""

__version__ = "0.1.0"

# Check for JAX support
import importlib.util
from typing import List, Dict, Any, Optional, Union, Callable


# Legacy imports (non-JAX versions)
from jaxabm.legacy import (
    Agent as LegacyAgent,
    Model as LegacyModel,
    Collector as LegacyCollector,
    Environment as LegacyEnvironment,
    Network as LegacyNetwork,
    Scheduler as LegacyScheduler,
    utils as legacy_utils
)


# Function to check if JAX is available
def has_jax() -> bool:
    """Check if JAX is available in the current environment.
    
    Returns:
        True if JAX is available, False otherwise
    """
    try:
        spec = importlib.util.find_spec("jax")
        return spec is not None
    except (ModuleNotFoundError, ImportError):
        return False


# Define what we'll export
__all__ = [
    # Core components
    "ModelConfig",
    "AgentType", 
    "AgentCollection",
    "Model",
    # Analysis tools
    "SensitivityAnalysis", 
    "ModelCalibrator",
    # Utility functions
    "convert_to_numpy",
    "format_time",
    "run_parallel_simulations",
    # Legacy components
    "LegacyAgent",
    "LegacyModel",
    # Status checks
    "has_jax",
    "jax_available"
]

# Initially set JAX components to None
AgentType = None
AgentCollection = None
Model = None
ModelConfig = None
SensitivityAnalysis = None
ModelCalibrator = None
# Initialize utility functions to None
convert_to_numpy = None
format_time = None
run_parallel_simulations = None

# Load components if JAX is available
HAS_JAX_LOADED = False
if has_jax():
    try:
        # Core components from their definitive locations
        from .core import ModelConfig 
        from .agent import AgentType, AgentCollection
        from .model import Model
        from .analysis import SensitivityAnalysis, ModelCalibrator
        # Import commonly used utilities
        from .utils import convert_to_numpy, format_time, run_parallel_simulations
        HAS_JAX_LOADED = True
    except ImportError as e:
        print(f"JaxABM warning: Could not import JAX components despite JAX being found. Error: {e}")
        HAS_JAX_LOADED = False


# --- Legacy Components --- 
# Keep the legacy imports as they were if needed
try:
    from jaxabm.legacy import (
        Agent as LegacyAgent,
        Model as LegacyModel,
        Collector as LegacyCollector,
        Environment as LegacyEnvironment,
        Network as LegacyNetwork,
        Scheduler as LegacyScheduler,
        utils as legacy_utils
    )
    # Add other legacy exports if needed
    # from jaxabm.legacy.space import ...
    # from jaxabm.legacy.batchrunner import batch_run
    # from jaxabm.legacy.visualization import JaxABMVisualization
except ImportError:
    print("JaxABM warning: Could not import legacy components.")
    # Define legacy placeholders if necessary
    LegacyAgent = None
    LegacyModel = None
    # ... other legacy placeholders ...

# Final check function for user convenience
def jax_available() -> bool:
    """Check if JAX components were successfully loaded.
    
    Returns:
        bool: True if JAX components were successfully loaded, False otherwise
    """
    return HAS_JAX_LOADED 