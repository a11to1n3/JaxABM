"""
Utility functions for the JAX-based agent-based modeling framework.

This module provides helper functions for common tasks in agent-based modeling,
such as data conversion, visualization, and validation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from jaxabm.model import Model
import jax
import jax.numpy as jnp
import numpy as np


def convert_to_numpy(data: Any) -> Any:
    """Convert JAX arrays to NumPy arrays recursively.
    
    This function handles nested dictionaries and lists containing JAX arrays,
    converting them to NumPy arrays for easier interoperability with other libraries.
    
    Args:
        data: JAX array, dictionary, list, or other data structure
        
    Returns:
        Equivalent structure with JAX arrays converted to NumPy arrays
    """
    if isinstance(data, dict):
        return {k: convert_to_numpy(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_numpy(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_to_numpy(item) for item in data)
    elif isinstance(data, jax.Array):
        return np.array(data)
    else:
        return data


def is_valid_params(params: Dict[str, Any], required_keys: List[str]) -> bool:
    """Check if parameters dictionary contains all required keys.
    
    Args:
        params: Parameters dictionary to check
        required_keys: List of keys that must be present
        
    Returns:
        True if all required keys are present, False otherwise
    """
    return all(key in params for key in required_keys)


def format_time(seconds: float) -> str:
    """Format time in seconds to a human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        String representation of time
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        seconds = seconds % 60
        return f"{minutes}m {seconds:.2f}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        seconds = seconds % 60
        return f"{hours}h {minutes}m {seconds:.2f}s"


def mean_over_runs(results_list: List[Dict[str, List[float]]]) -> Dict[str, List[float]]:
    """Calculate mean values over multiple simulation runs.
    
    Args:
        results_list: List of results dictionaries from multiple runs
        
    Returns:
        Dictionary with mean values for each metric
    """
    if not results_list:
        return {}
    
    # Get all keys from the first results dict
    keys = results_list[0].keys()
    
    # Create a dictionary to store the results
    mean_results = {}
    
    for key in keys:
        # Check if all result dictionaries have this key and the lists have the same length
        if all(key in results and len(results[key]) == len(results_list[0][key]) 
               for results in results_list):
            # Convert all values to NumPy arrays for easier computation
            arrays = [np.array(results[key]) for results in results_list]
            # Calculate the mean
            mean_results[key] = np.mean(arrays, axis=0).tolist()
    
    return mean_results


def standardize_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Standardize metrics by converting any non-float values.
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        Dictionary with all metrics as float values
    """
    result = {}
    for key, value in metrics.items():
        if hasattr(value, "item"):
            # Convert JAX or NumPy arrays to Python floats
            result[key] = float(value.item())
        elif isinstance(value, (int, float)):
            result[key] = float(value)
        else:
            # Skip non-numeric values
            continue
    return result


def run_parallel_simulations(
    model_factory: Callable[..., 'Model'], 
    param_sets: List[Dict[str, Any]], 
    num_runs: int = 1,
    seed_offset: int = 0
) -> List[Dict[str, Any]]:
    """Run multiple simulations in parallel with different parameter sets.
    
    This function uses JAX's parallelization capabilities to run multiple simulations
    with different parameter sets in parallel.
    
    Args:
        model_factory: Function that creates a model instance
        param_sets: List of parameter dictionaries for different simulation runs
        num_runs: Number of runs for each parameter set
        seed_offset: Starting offset for random seeds
        
    Returns:
        List of results dictionaries from all simulation runs
    """
    from jaxabm.core import ModelConfig
    
    all_results = []
    
    # Function to run a single simulation
    def run_single_sim(params, seed):
        config = ModelConfig(seed=seed)
        # Create model using factory function
        model = model_factory(params=params, config=config)
        # Run the model and get results
        return model.run()
    
    # Iterate over parameter sets
    for i, params in enumerate(param_sets):
        # Run multiple times with different seeds
        for j in range(num_runs):
            seed = seed_offset + i * num_runs + j
            # Run the simulation
            print(f"Running simulation {i+1}/{len(param_sets)}, run {j+1}/{num_runs}, seed={seed}")
            try:
                results = run_single_sim(params, seed)
                # Add parameter info to results
                results['params'] = params
                results['seed'] = seed
                all_results.append(results)
            except Exception as e:
                print(f"Error in simulation {i+1}/{len(param_sets)}, run {j+1}/{num_runs}: {e}")
    
    return all_results 