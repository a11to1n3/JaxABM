"""
JaxABM: An agent-based modeling framework accelerated by JAX.

This package provides tools for building, running, and analyzing agent-based models
with an easy-to-use AgentPy-like interface while maintaining the performance benefits
of JAX acceleration.

The framework now provides primarily an AgentPy-like interface:
- Agent: Base class for creating agents
- AgentList: Container for managing collections of agents
- Environment: Container for environment state and spatial structures
- Grid: Grid environment for spatial models
- Network: Network environment for network models
- Model: Base class for creating models
- Results: Container for simulation results

The core JAX components are still available:
- AgentType: Protocol for defining agent behaviors
- AgentCollection: Collection of agents of the same type
- ModelConfig: Configuration for model execution
- JaxModel: Core model implementation with JAX acceleration

The legacy version (non-JAX) is available directly from the package root.
The JAX-accelerated version is available when JAX is installed.
"""

__version__ = "0.1.0"

# Check for JAX support
import importlib.util
from typing import List, Dict, Any, Optional, Union, Callable, Type


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
    # AgentPy-like components (primary API)
    "Agent",
    "AgentList",
    "Environment",
    "Grid",
    "Network",
    "Model",
    "Results",
    # Original JaxABM components
    "AgentType", 
    "AgentCollection",
    "JaxModel",
    "ModelConfig",
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
# AgentPy-like components
Agent = None
AgentList = None
Environment = None
Grid = None
Network = None
Model = None
Results = None
# Original JaxABM components
AgentType = None
AgentCollection = None
JaxModel = None
ModelConfig = None
# Analysis tools
SensitivityAnalysis = None
ModelCalibrator = None
# Utility functions
convert_to_numpy = None
format_time = None
run_parallel_simulations = None

# Load components if JAX is available
HAS_JAX_LOADED = False
if has_jax():
    try:
        # Import core JaxABM components
        from .core import ModelConfig 
        from .agent import AgentType, AgentCollection
        from .model import Model as JaxModel
        from .analysis import SensitivityAnalysis, ModelCalibrator
        
        # Import utility functions
        from .utils import convert_to_numpy, format_time, run_parallel_simulations
        
        # Import AgentPy-like components (primary API)
        from .agentpy import (
            Agent, 
            AgentList, 
            Environment, 
            Grid,
            Network,
            Model, 
            Results
        )
        
        # Set flag
        HAS_JAX_LOADED = True
    except ImportError as e:
        print(f"JaxABM warning: Could not import JAX components despite JAX being found. Error: {e}")
        HAS_JAX_LOADED = False
else:
    print("JaxABM warning: JAX not found. Using legacy components.")


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
except ImportError:
    print("JaxABM warning: Could not import legacy components.")
    # Define legacy placeholders if necessary
    LegacyAgent = None
    LegacyModel = None


# Final check function for user convenience
def jax_available() -> bool:
    """Check if JAX components were successfully loaded.
    
    Returns:
        bool: True if JAX components were successfully loaded, False otherwise
    """
    return HAS_JAX_LOADED 