"""
Basic example demonstrating the core agent abstractions in JaxABM.

This example creates a simple agent-based model using the core JaxABM abstractions:
- AgentType: Defines agent behavior (init and update functions)
- AgentCollection: Manages a collection of agents of the same type
- ModelConfig: Configures model simulation parameters
- Model: Coordinates agent collections and simulation execution
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

# Add parent directory to path to allow imports from jaxabm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jax import random

from jaxabm.agent import AgentType, AgentCollection
from jaxabm.core import ModelConfig
from jaxabm.model import Model


class RandomWalker(AgentType):
    """Simple agent that performs a random walk in 2D space."""
    
    def __init__(self, init_scale=1.0, step_size=0.1):
        """Initialize RandomWalker with specific parameters.
        
        Args:
            init_scale: Scale for initial positions
            step_size: Size of random steps
        """
        self.init_scale = init_scale
        self.step_size = step_size

    def init_state(self, model_config: ModelConfig, key: jax.Array) -> Dict[str, Any]:
        """Initialize agent state with random position.
        
        Args:
            model_config: Model configuration (contains seed)
            key: JAX random key for this agent
            
        Returns:
            Initial agent state dictionary
        """
        # Use parameters defined in the instance
        
        key1, key2 = random.split(key)
        pos_x = random.uniform(key1, minval=-self.init_scale, maxval=self.init_scale)
        pos_y = random.uniform(key2, minval=-self.init_scale, maxval=self.init_scale)
        
        return {
            'position_x': pos_x,
            'position_y': pos_y,
            'distance': jnp.sqrt(pos_x**2 + pos_y**2),
            'step_count': jnp.array(0, dtype=jnp.int32)
        }
    
    def update(self, state: Dict[str, Any], model_state: Dict[str, Any], 
               model_config: ModelConfig, key: jax.Array) -> Dict[str, Any]: # Returns only state
        """Update agent position with a random step.
        
        Args:
            state: Current agent state dictionary
            model_state: Current full model state (environment + all agent states)
            model_config: Model configuration
            key: JAX random key for this agent
            
        Returns:
            Updated agent state dictionary
        """
        # Use step_size from class attribute
        
        # Generate random step
        key1, key2 = random.split(key)
        step_x = random.uniform(key1, minval=-self.step_size, maxval=self.step_size)
        step_y = random.uniform(key2, minval=-self.step_size, maxval=self.step_size)
        
        # Update position
        new_x = state['position_x'] + step_x
        new_y = state['position_y'] + step_y
        
        # Calculate new distance from origin
        new_distance = jnp.sqrt(new_x**2 + new_y**2)
        
        # Create and return the new state dictionary
        new_state = {
            'position_x': new_x,
            'position_y': new_y,
            'distance': new_distance,
            'step_count': state['step_count'] + 1
        }
        
        # No outputs returned directly, state contains the updated info
        return new_state


def update_model_state(env_state: Dict[str, Any], agent_states: Dict[str, Dict[str, Any]], 
                       params: Dict[str, Any], key: jax.Array) -> Dict[str, Any]: # Updated signature
    """Update the model environment state based on agent states.
    
    Args:
        env_state: Current environment state dictionary.
        agent_states: Dictionary mapping collection names to their state dictionaries.
                      Example: {'walkers': {'position_x': Array([...]), ...}}
        params: Model parameters (if any).
        key: JAX random key for this update step.
        
    Returns:
        Updated environment state dictionary.
    """
    # Extract walkers' states
    walker_states = agent_states.get('walkers')
    
    # Calculate statistics from walker states if available
    if walker_states:
        mean_x = jnp.mean(walker_states.get('position_x', jnp.array(0.0)))
        mean_y = jnp.mean(walker_states.get('position_y', jnp.array(0.0)))
        std_x = jnp.std(walker_states.get('position_x', jnp.array(0.0)))
        std_y = jnp.std(walker_states.get('position_y', jnp.array(0.0)))
        mean_distance = jnp.mean(walker_states.get('distance', jnp.array(0.0)))
    else:
        mean_x, mean_y, std_x, std_y, mean_distance = [jnp.array(0.0)] * 5
    
    # Update environment state with calculated statistics
    new_env_state = {
        'mean_position_x': mean_x,
        'mean_position_y': mean_y,
        'std_position_x': std_x,
        'std_position_y': std_y,
        'mean_distance': mean_distance,
        'time': env_state.get('time', 0) + 1
    }
    
    return new_env_state


def compute_metrics(env_state: Dict[str, Any], agent_states: Dict[str, Dict[str, Any]], 
                    params: Dict[str, Any]) -> Dict[str, Any]: # Updated signature
    """Compute model metrics from the current environment state.
    
    Args:
        env_state: Current environment state dictionary.
        agent_states: Dictionary of agent states by collection (unused here, but part of signature).
        params: Model parameters (unused here).
        
    Returns:
        Dictionary of computed metrics.
    """
    # Metrics are derived directly from the environment state calculated in update_model_state
    return {
        'mean_x': env_state['mean_position_x'],
        'mean_y': env_state['mean_position_y'],
        'std_x': env_state['std_position_x'],
        'std_y': env_state['std_position_y'],
        'mean_distance': env_state['mean_distance']
    }


def create_random_walk_model(num_walkers=100, init_scale=1.0, step_size=0.1, seed=0):
    """Create a random walk model.
    
    Args:
        num_walkers: Number of walker agents
        init_scale: Scale for initial positions
        step_size: Size of random steps
        seed: Random seed
        
    Returns:
        Configured model instance
    """
    # Create walker agent type instance with specific parameters
    # Parameters are now part of the agent type definition or passed via model params
    walker_agent_type = RandomWalker(init_scale=init_scale, step_size=step_size)
    
    # Create agent collection using the new signature
    walkers = AgentCollection(
        agent_type=walker_agent_type,
        num_agents=num_walkers
    )
    
    # Define initial environment state variables
    initial_env_state = {
        'mean_position_x': jnp.array(0.0),
        'mean_position_y': jnp.array(0.0),
        'std_position_x': jnp.array(init_scale / jnp.sqrt(3)),
        'std_position_y': jnp.array(init_scale / jnp.sqrt(3)),
        'mean_distance': jnp.array(0.0),
        'time': jnp.array(0, dtype=jnp.int32) # Ensure dtype for JAX
    }
    
    # Create model config
    config = ModelConfig(
        steps=100, 
        collect_interval=1,
        seed=seed
    )
    
    # Create model instance using the new signature
    # Pass model-level parameters if any (none in this case)
    model_params = {} 
    model = Model(
        params=model_params,
        config=config,
        update_state_fn=update_model_state,
        metrics_fn=compute_metrics
    )

    # Add agent collection to the model
    model.add_agent_collection('walkers', walkers)

    # Add initial environment state variables to the model
    for name, value in initial_env_state.items():
        model.add_env_state(name, value)
    
    # Initialization is handled by model.run(), no need for model.initialize() here
    
    return model


def plot_results(results):
    """Plot the metrics from model simulation.
    
    Args:
        results: Dictionary of metrics from model.run()
    """
    # Attempt to use a nicer style if available
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        pass
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot mean position
    ax1.plot(results['mean_x'], results['mean_y'], 'b-', label='Mean position')
    ax1.scatter(results['mean_x'][0], results['mean_y'][0], color='green', s=100, label='Start')
    ax1.scatter(results['mean_x'][-1], results['mean_y'][-1], color='red', s=100, label='End')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('Mean Walker Position')
    ax1.grid(True)
    ax1.legend()
    
    # Plot standard deviation and mean distance
    steps = list(range(len(results['std_x'])))
    ax2.plot(steps, results['std_x'], 'r-', label='Std Dev X')
    ax2.plot(steps, results['std_y'], 'b-', label='Std Dev Y')
    ax2.plot(steps, results['mean_distance'], 'g-', label='Mean Distance')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Value')
    ax2.set_title('Distribution Statistics')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def main():
    """Run the random walk example."""
    print("Running random walk example with JaxABM...")
    
    # Create and run model
    model = create_random_walk_model(
        num_walkers=1000,  # More walkers for better statistics
        init_scale=1.0,
        step_size=0.1,
        seed=42
    )
    
    # Run simulation
    results = model.run()
    
    # Print final metrics
    print("\nFinal metrics:")
    for metric, values in results.items():
        print(f"  {metric}: {values[-1]:.4f}")
    
    # Plot results
    try:
        plot_results(results)
    except Exception as e:
        print(f"Error plotting results: {e}")
    
    print("\nExample completed.")


if __name__ == "__main__":
    main() 