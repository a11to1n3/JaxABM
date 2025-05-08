"""
Core functionality of the JaxABM agent-based modeling framework.

This module serves as the main entry point for the framework, providing access
to the key components for building and running agent-based models with JAX acceleration.

MIGRATION GUIDE:
Previously, all components were in a single monolithic file. Now, components have been
organized into separate modules for better maintainability:

- core.py:     Core configuration and framework definitions (ModelConfig)
- agent.py:    Agent-related classes (AgentType, AgentCollection)
- model.py:    Model simulation class (Model)
- analysis.py: Analysis tools (SensitivityAnalysis, ModelCalibrator)
- utils.py:    Utility functions (convert_to_numpy, format_time, etc.)

If you had imported from "jaxabm.core" previously, you should now import from the
appropriate module. The package's __init__.py provides all components as top-level imports 
for backward compatibility.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, List, Tuple, Protocol, Callable, Optional, Union

# This module focuses on core configuration components 
# and provides a clean separation of concerns

# Check for JAX availability
def has_jax():
    """Check if JAX is available.
    
    Returns:
        True if JAX is available, False otherwise
    """
    try:
        import jax
        return True
    except ImportError:
        return False

# Legacy support (will be deprecated in future versions)
try:
    from jaxabm.legacy import (
        Model as LegacyModel,
        Agent as LegacyAgent,
        AgentSet as LegacyAgentSet,
        DataCollector as LegacyDataCollector
    )
except ImportError:
    pass

# --- Core Configuration ---

class ModelConfig:
    """Configuration for model execution.
    
    This class holds configuration parameters for model execution, including
    random seed, number of steps, and history tracking options.
    
    Attributes:
        seed: Random seed for reproducibility
        steps: Number of simulation steps to run
        track_history: Whether to track model history
        collect_interval: Interval for collecting history (every N steps)
    """
    
    def __init__(
        self, 
        seed: int = 0, 
        steps: int = 100,
        track_history: bool = True,
        collect_interval: int = 1
    ):
        """Initialize model configuration.
        
        Args:
            seed: Random seed for reproducibility
            steps: Number of simulation steps to run 
            track_history: Whether to track model history
            collect_interval: Interval for collecting history (every N steps)
        """
        self.seed = seed
        self.steps = steps
        self.track_history = track_history
        self.collect_interval = collect_interval

# Function to show framework info
def show_info():
    """Display information about the JaxABM framework.
    
    Prints version information, available components, and JAX status.
    """
    from jaxabm import __version__
    
    print(f"JaxABM v{__version__}")
    print("Agent-based modeling framework with JAX acceleration")
    print()
    
    if has_jax():
        import jax
        print(f"JAX version: {jax.__version__}")
        print(f"Devices available: {jax.devices()}")
        print("JAX-accelerated components available")
    else:
        print("JAX not found. Only legacy components available.")
        print("Install JAX for acceleration capabilities.")
    
    print()
    print("Available components:")
    print("  - Model: Main simulation class")
    print("  - AgentCollection: Collection of agents of the same type")
    print("  - AgentType: Protocol for defining agent behavior")
    print("  - SensitivityAnalysis: Analysis of parameter sensitivity")
    print("  - ModelCalibrator: Parameter calibration tools")
    print()
    print("For more information, visit: https://github.com/jaxabm")
