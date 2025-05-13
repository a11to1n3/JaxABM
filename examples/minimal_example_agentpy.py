"""
Minimal example using the AgentPy-like interface of the JaxABM framework.

This script demonstrates the same random walk model as minimal_example.py,
but using the new AgentPy-like interface.

This script shows how to:
1. Define an agent class with setup and step methods
2. Create a model class with setup and step methods
3. Add agents to the model using add_agents
4. Run the model and analyze results
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jaxabm as jx
import jax
import jax.numpy as jnp
from jax import random


class RandomWalker(jx.Agent):
    """Simple agent that performs a random walk."""
    
    def setup(self):
        """Initialize agent with a random position."""
        # Parameters are available via self.p
        init_scale = self.p.get('init_scale', 1.0)
        
        # Use JAX's random generator for initial position
        # Note: In a real model, we would use a better way to get random keys
        key = random.PRNGKey(0)
        
        return {
            'position': random.normal(key, (2,)) * init_scale
        }
    
    def step(self, model_state):
        """Take a random step."""
        # Get parameters from agent
        step_size = self.p.get('step_size', 0.1)
        
        # Get current position
        position = self._state['position']
        
        # Generate random movement
        # Note: In a real model, we would use a better way to get random keys
        key = random.PRNGKey(0)
        movement = random.normal(key, (2,)) * step_size
        
        # Update position
        new_position = position + movement
        
        # Apply boundary condition if available
        if model_state and 'env' in model_state and 'bounds' in model_state['env']:
            bounds = model_state['env']['bounds']
            new_position = jnp.clip(new_position, bounds[0], bounds[1])
        
        # Return new state
        return {'position': new_position}


class RandomWalkModel(jx.Model):
    """Model with random walking agents."""
    
    def setup(self):
        """Set up model with agents and environment."""
        # Add agents
        self.walkers = self.add_agents(
            n=self.p.get('num_agents', 1000),
            agent_class=RandomWalker,
            name='walkers',
            init_scale=self.p.get('init_scale', 1.0),
            step_size=self.p.get('step_size', 0.1)
        )
        
        # Add environment variables
        self.env.add_state('bounds', jnp.array([-10.0, 10.0]))
        self.env.add_state('mean_position', jnp.zeros(2))
        self.env.add_state('std_position', jnp.ones(2))
        self.env.add_state('min_position', -jnp.ones(2))
        self.env.add_state('max_position', jnp.ones(2))
    
    def update_state(self, env_state, agent_states, model_params, key):
        """Update model environment state based on agent states."""
        # Access walker states
        walker_states = agent_states.get('walkers')
        
        if not walker_states:
            return env_state  # No walkers, return current state
        
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
    
    def compute_metrics(self, env_state, agent_states, model_params):
        """Compute metrics from model state."""
        return {
            'mean_x': env_state['mean_position'][0],
            'mean_y': env_state['mean_position'][1],
            'std_x': env_state['std_position'][0],
            'std_y': env_state['std_position'][1],
        }


def run_simulation(num_agents=1000, num_steps=10):
    """Run a simulation and display results."""
    print(f"Running simulation with {num_agents} agents for {num_steps} steps...")
    
    # Create parameters
    parameters = {
        'num_agents': num_agents,
        'steps': num_steps,
        'init_scale': 1.0,
        'step_size': 0.1,
        'seed': 42
    }
    
    # Create and run model
    model = RandomWalkModel(parameters)
    results = model.run()
    
    # Print results
    print("\nSimulation completed. Results:")
    
    # Access results through the results object
    if hasattr(results, '_data'):
        steps = len(results._data.get('step', []))
        print(f"  Steps: {steps}")
        
        mean_x = results._data.get('mean_x', [])
        mean_y = results._data.get('mean_y', [])
        std_x = results._data.get('std_x', [])
        std_y = results._data.get('std_y', [])
        
        if mean_x and mean_y and std_x and std_y:
            print(f"  Final mean position: ({float(mean_x[-1]):.4f}, {float(mean_y[-1]):.4f})")
            print(f"  Final position std: ({float(std_x[-1]):.4f}, {float(std_y[-1]):.4f})")
        
        # Plot results
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            ax1.plot(mean_x, mean_y, 'b-')
            ax1.set_xlabel('Mean X Position')
            ax1.set_ylabel('Mean Y Position')
            ax1.set_title('Mean Agent Position')
            ax1.grid(True)
            
            ax2.plot(results._data.get('step', []), std_x, 'r-', label='X')
            ax2.plot(results._data.get('step', []), std_y, 'g-', label='Y')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Standard Deviation')
            ax2.set_title('Position Standard Deviation')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig('random_walk_results.png')
            print("Results plotted and saved to 'random_walk_results.png'.")
        except Exception as e:
            print(f"Error plotting results: {e}")
    
    # Verify simulation completed successfully
    if steps == num_steps:
        print("\nSimulation completed successfully!")
    
    return results


if __name__ == "__main__":
    run_simulation()