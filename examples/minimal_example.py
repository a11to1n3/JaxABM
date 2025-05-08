"""
Minimal example demonstrating the core functionality of the JaxABM framework.

This script shows how to:
1. Define an agent type with initialization and update behaviors
2. Create an agent collection
3. Configure and run a model
4. Process and analyze results
"""

import os
import sys
import numpy as np
from functools import partial
from typing import Dict, Any

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jax import random

# Import components from their correct modules after refactoring
from jaxabm.agent import AgentType, AgentCollection
from jaxabm.core import ModelConfig
from jaxabm.model import Model


class RandomWalker(AgentType):
    """Simple agent that performs a random walk."""
    
    def __init__(self, init_scale=1.0, step_size=0.1):
        """Initialize RandomWalker agent type.
        
        Args:
            init_scale: Scale for initial position
            step_size: Size of random steps
        """
        self.init_scale = init_scale
        self.step_size = step_size

    def init_state(self, model_config: ModelConfig, key: jax.Array) -> Dict[str, Any]:
        """Initialize agent with a random position."""
        return {
            'position': random.normal(key, (2,)) * self.init_scale
        }
    
    def update(self, state: Dict[str, Any], model_state: Dict[str, Any], 
               model_config: ModelConfig, key: jax.Array) -> Dict[str, Any]:
        """Take a random step."""
        # Generate random movement
        movement = random.normal(key, (2,)) * self.step_size
        
        # Update position
        new_position = state['position'] + movement
        
        # Boundary condition (optional) - Access env state
        if 'bounds' in model_state['env']:
            bounds = model_state['env']['bounds']
            new_position = jnp.clip(new_position, bounds[0], bounds[1])
        
        # Return new state only
        new_state = {'position': new_position}
        # Outputs are no longer returned, step size could be stored in state if needed
        return new_state


def update_state(env_state: Dict[str, Any], agent_states: Dict[str, Dict[str, Any]], 
                 params: Dict[str, Any], key: jax.Array) -> Dict[str, Any]:
    """Update model environment state based on agent states."""
    # Access walker states
    walker_states = agent_states.get('walkers')
    
    if not walker_states:
        return env_state # No walkers, return current state

    positions = walker_states['position']
    
    # Calculate statistics
    mean_pos = jnp.mean(positions, axis=0)
    std_pos = jnp.std(positions, axis=0)
    min_pos = jnp.min(positions, axis=0)
    max_pos = jnp.max(positions, axis=0)
    
    # Create new environment state
    new_env_state = {
        'mean_position': mean_pos,
        'std_position': std_pos,
        'min_position': min_pos,
        'max_position': max_pos,
        'bounds': env_state['bounds']  # Keep bounds unchanged
    }
    
    return new_env_state


def compute_metrics(env_state: Dict[str, Any], agent_states: Dict[str, Dict[str, Any]], 
                    params: Dict[str, Any]) -> Dict[str, Any]:
    """Compute metrics from model state."""
    # Metrics now come from env_state, step size is not available unless put in agent state
    # Re-calculate mean step size here if needed, or remove it.
    # Let's remove it for simplicity, as AgentCollection.update no longer returns outputs.
    
    return {
        'mean_x': env_state['mean_position'][0],
        'mean_y': env_state['mean_position'][1],
        'std_x': env_state['std_position'][0],
        'std_y': env_state['std_position'][1],
        # 'mean_step_size': jnp.mean(step_sizes), # Removed
        # 'max_step_size': jnp.max(step_sizes)   # Removed
    }


def run_simulation(num_agents=1000, num_steps=10):
    """Run a simulation and display results."""
    print(f"Running simulation with {num_agents} agents for {num_steps} steps...")
    
    # Create AgentType instance
    walker_agent_type = RandomWalker(init_scale=1.0, step_size=0.1)

    # Create agent collection
    walkers = AgentCollection(
        agent_type=walker_agent_type,
        num_agents=num_agents
    )
    
    # Create initial environment state
    initial_env_state = {
        'mean_position': jnp.zeros(2),
        'std_position': jnp.ones(2),
        'min_position': -jnp.ones(2),
        'max_position': jnp.ones(2),
        'bounds': jnp.array([-10.0, 10.0])
    }
    
    # Create model parameters (empty for this example)
    model_params = {}
    
    # Create model config
    config = ModelConfig(
        steps=num_steps,
        collect_interval=1,
        seed=42
    )
    
    # Create model
    model = Model(
        params=model_params,
        config=config,
        update_state_fn=update_state,
        metrics_fn=compute_metrics
    )
    
    # Add collection and initial state
    model.add_agent_collection('walkers', walkers)
    for name, value in initial_env_state.items():
        model.add_env_state(name, value)

    # Run model
    results = model.run()
    
    # Print results
    print("\nSimulation completed. Results:")
    print(f"  Steps: {len(results['step'])}")
    print(f"  Final mean position: ({float(results['mean_x'][-1]):.4f}, {float(results['mean_y'][-1]):.4f})")
    print(f"  Final position std: ({float(results['std_x'][-1]):.4f}, {float(results['std_y'][-1]):.4f})")
    # print(f"  Mean step size: {float(results['mean_step_size'][-1]):.4f}") # Step size metric removed
    
    # Verify simulation completed successfully
    if len(results['step']) == num_steps:
        print("\nSimulation completed successfully!")
    return results


if __name__ == "__main__":
    run_simulation()