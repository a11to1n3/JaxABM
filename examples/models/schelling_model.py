"""
Schelling Segregation Model using JaxABM with AgentPy-like interface.

This example demonstrates how to implement a classic agent-based model using
the new AgentPy-like interface for JaxABM. The model shows how spatial segregation
can emerge from simple individual preferences.
"""

import jaxabm as jx
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
import time


class SchellingSocialAgent(jx.Agent):
    """Agent for the Schelling segregation model."""
    
    def setup(self):
        """Initialize agent state.
        
        This method is called once during model initialization.
        """
        # Type is either 0 or 1 (representing two social groups)
        return {
            'type': 0,             # Default type (will be set by model)
            'position': jnp.zeros(2, dtype=jnp.int32),  # (x, y) position
            'satisfied': False,    # Whether agent is satisfied with neighborhood
            'moves': 0             # Number of times agent has moved
        }
    
    def step(self, model_state):
        """Execute agent behavior.
        
        This method is called at each time step.
        """
        # Get agent's current state
        agent_type = self._state['type']
        position = self._state['position']
        moves = self._state['moves']
        
        # Get environment state from model
        env_state = model_state.get('env', {})
        
        # Extract grid information
        grid_state = env_state.get('grid', jnp.zeros((10, 10), dtype=jnp.int32))
        empty_cells = env_state.get('empty_cells', jnp.zeros((0, 2), dtype=jnp.int32))
        
        # Get similarity threshold from model parameters
        sim_threshold = model_state.get('params', {}).get('similarity_threshold', 0.5)
        
        # Check if agent is satisfied based on neighborhood similarity
        # In a real model, we would implement this check with JAX operations
        # For simplicity, we'll use a placeholder
        satisfied = jnp.array([True])  # Placeholder
        
        # Move if not satisfied - in a real model, we would implement the move
        # For simplicity, we'll use a placeholder for the new position
        new_position = position
        new_moves = moves
        
        # Return updated state
        return {
            'type': agent_type,
            'position': new_position,
            'satisfied': satisfied[0],
            'moves': new_moves
        }


class SchellingModel(jx.Model):
    """Schelling segregation model with two types of agents on a grid."""
    
    def setup(self):
        """Set up model with agents and environment."""
        # Create grid
        grid_size = self.p.get('grid_size', 20)
        self.grid = jx.Grid(self, (grid_size, grid_size))
        
        # Create agents
        n_agents = self.p.get('n_agents', 300)
        ratio = self.p.get('ratio', 0.5)  # Ratio of type 0 to 1
        
        # Initialize all agents
        self.agents = self.add_agents(n_agents, SchellingSocialAgent)
        
        # Set agent types
        n_type0 = int(n_agents * ratio)
        agent_types = jnp.concatenate([
            jnp.zeros(n_type0, dtype=jnp.int32),
            jnp.ones(n_agents - n_type0, dtype=jnp.int32)
        ])
        
        # Position agents randomly on grid (avoiding overlaps)
        positions = self._get_random_positions(n_agents, grid_size)
        
        # Add types to agent collection
        if hasattr(self.agents.collection, '_states') and self.agents.collection._states is not None:
            # Initialize agent states with types
            self.agents.collection._states['type'] = agent_types
            self.agents.collection._states['position'] = positions
            if 'satisfied' not in self.agents.collection._states:
                self.agents.collection._states['satisfied'] = jnp.zeros(n_agents, dtype=bool)
            if 'moves' not in self.agents.collection._states:
                self.agents.collection._states['moves'] = jnp.zeros(n_agents, dtype=jnp.int32)
        else:
            # Initialize states if they don't exist
            if hasattr(self.agents.collection, '_states'):
                self.agents.collection._states = {
                    'type': agent_types,
                    'position': positions,
                    'satisfied': jnp.zeros(n_agents, dtype=bool),
                    'moves': jnp.zeros(n_agents, dtype=jnp.int32)
                }
        
        # Initialize grid state - represents what type of agent is at each position
        # -1 means empty, 0 or 1 is the agent type
        grid_state = -jnp.ones((grid_size, grid_size), dtype=jnp.int32)
        
        # Add agents to grid state
        # In a real implementation, we would do this with JAX operations
        # For demonstration purposes, we'll convert to numpy and back
        grid_np = np.array(grid_state)
        positions_np = np.array(positions)
        for i in range(n_agents):
            x, y = positions_np[i]
            grid_np[x, y] = np.array(agent_types)[i]
        
        # Convert back to JAX array
        grid_state = jnp.array(grid_np)
        
        # Calculate initial empty cells
        empty_mask = grid_state == -1
        empty_indices = jnp.array(np.column_stack(np.where(empty_mask)))
        
        # Add grid and empty cells to environment state
        self.env.add_state('grid', grid_state)
        self.env.add_state('empty_cells', empty_indices)
        
        # Initialize metrics
        self.env.add_state('segregation_index', 0.0)
        self.env.add_state('percent_satisfied', 0.0)
        self.env.add_state('total_moves', 0)
    
    def _get_random_positions(self, n_agents, grid_size):
        """Generate random positions for agents.
        
        Positions are guaranteed to be unique (no overlaps).
        
        Args:
            n_agents: Number of agents.
            grid_size: Size of the grid.
            
        Returns:
            Array of positions with shape (n_agents, 2).
        """
        # For simplicity, generate positions without JAX
        # In a real model, we would use JAX operations
        rng = np.random.RandomState(self.p.get('seed', 42))
        
        # Generate all possible positions
        all_positions = np.array([(x, y) for x in range(grid_size) for y in range(grid_size)])
        
        # Shuffle and select n_agents positions
        rng.shuffle(all_positions)
        selected_positions = all_positions[:n_agents]
        
        # Convert to JAX array
        return jnp.array(selected_positions, dtype=jnp.int32)
    
    def compute_metrics(self, env_state, agent_states, model_params):
        """Compute model metrics."""
        # Get agent states
        agent_types = agent_states.get('agents', {}).get('type', jnp.array([]))
        agent_satisfied = agent_states.get('agents', {}).get('satisfied', jnp.array([]))
        agent_moves = agent_states.get('agents', {}).get('moves', jnp.array([]))
        
        # Compute percentage of satisfied agents
        if len(agent_satisfied) > 0:
            percent_satisfied = jnp.mean(agent_satisfied)
        else:
            percent_satisfied = jnp.array(0.0)
        
        # Compute total moves
        total_moves = jnp.sum(agent_moves)
        
        # Compute segregation index (placeholder - in a real model we would implement this)
        segregation_index = jnp.array(0.0)
        
        # Return metrics
        return {
            'percent_satisfied': percent_satisfied,
            'segregation_index': segregation_index,
            'total_moves': total_moves
        }
    
    def update_state(self, env_state, agent_states, model_params, key):
        """Update environment state based on agent states."""
        # Update environment state with metrics
        new_env_state = dict(env_state)
        
        # Update grid state based on agent positions (placeholder)
        # In a real model, we would implement this with JAX operations
        
        return new_env_state


def run_schelling_model():
    """Run Schelling segregation model and visualize results."""
    # Define parameters
    parameters = {
        'grid_size': 20,
        'n_agents': 300,
        'ratio': 0.5,  # Ratio of type 0 to type 1 agents
        'similarity_threshold': 0.5,  # Minimum ratio of similar neighbors
        'steps': 20,
        'seed': 42
    }
    
    # Create and run model
    print("Creating model...")
    model = SchellingModel(parameters)
    
    print("Running simulation...")
    start_time = time.time()
    results = model.run()
    end_time = time.time()
    
    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot satisfaction over time
    if 'percent_satisfied' in results._data:
        axes[0].plot(results._data['percent_satisfied'], 'b-')
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Percentage Satisfied')
        axes[0].set_title('Agent Satisfaction')
        axes[0].grid(True)
    
    # Plot segregation index over time
    if 'segregation_index' in results._data:
        axes[1].plot(results._data['segregation_index'], 'r-')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Segregation Index')
        axes[1].set_title('Segregation')
        axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('schelling_results.png')
    print("Results plotted and saved to 'schelling_results.png'.")
    
    # Print final metrics
    if 'percent_satisfied' in results._data:
        print(f"Final percentage satisfied: {float(results._data['percent_satisfied'][-1]):.2%}")
    if 'total_moves' in results._data:
        print(f"Total moves: {int(results._data['total_moves'][-1])}")
    
    return results


if __name__ == "__main__":
    run_schelling_model()