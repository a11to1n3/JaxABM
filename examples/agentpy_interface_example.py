"""
AgentPy-like interface example for JaxABM.

This example shows how to use the new AgentPy-like interface for creating
and running agent-based models with JaxABM.
"""

import jaxabm as jx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time

# Define an agent class
class MyAgent(jx.Agent):
    """A simple agent with x and y position."""
    
    def setup(self):
        """Initialize agent state."""
        # Initialize with random position between 0 and 1
        return {
            'x': 0.5,
            'y': 0.5,
            'dx': 0.01,
            'dy': 0.01
        }
    
    def step(self, model_state):
        """Update agent state."""
        # Get current state
        x = self._state['x']
        y = self._state['y']
        dx = self._state['dx']
        dy = self._state['dy']
        
        # Move agents
        x += dx
        y += dy
        
        # Bounce off walls
        if x <= 0 or x >= 1:
            dx = -dx
        if y <= 0 or y >= 1:
            dy = -dy
        
        # Return updated state
        return {
            'x': x,
            'y': y,
            'dx': dx,
            'dy': dy
        }

# Define a model class
class BouncingAgentsModel(jx.Model):
    """A simple model with bouncing agents."""
    
    def setup(self):
        """Set up model with agents and environment."""
        # Add agents
        self.agents = self.add_agents(
            n=self.p.get('n_agents', 10),
            agent_class=MyAgent
        )
        
        # Add environment variables
        self.env.add_state('time', 0)
    
    def step(self):
        """Execute a single time step."""
        # Update environment variables
        if hasattr(self._model, 'state'):
            self._model.add_env_state('time', self._model.state['env'].get('time', 0) + 1)
        
        # Step all agents
        # The Model.step method already handles this, but we can add additional logic here
        
        # Record metrics
        if hasattr(self, '_model') and self._model.state:
            agent_states = self._model.state['agents'].get('myagents', {})
            if 'x' in agent_states and 'y' in agent_states:
                # Calculate average position
                avg_x = jnp.mean(agent_states['x'])
                avg_y = jnp.mean(agent_states['y'])
                
                # Record average position
                self.record('avg_x', float(avg_x))
                self.record('avg_y', float(avg_y))
                
                # Record distance from center
                distances = jnp.sqrt((agent_states['x'] - 0.5)**2 + (agent_states['y'] - 0.5)**2)
                avg_distance = jnp.mean(distances)
                self.record('avg_distance', float(avg_distance))


def run_simulation():
    """Run a simulation using the AgentPy-like interface."""
    # Define parameters
    parameters = {
        'n_agents': 100,
        'steps': 100,
        'seed': 42
    }
    
    # Create and run model
    print("Creating model...")
    model = BouncingAgentsModel(parameters)
    
    print("Running simulation...")
    start_time = time.time()
    results = model.run()
    end_time = time.time()
    
    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")
    
    # Plot results
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot average position over time
    if hasattr(results, 'variables'):
        try:
            ax[0].plot(results._data.get('avg_x', []), results._data.get('avg_y', []), 'b-')
            ax[0].set_xlabel('Average X Position')
            ax[0].set_ylabel('Average Y Position')
            ax[0].set_title('Agent Movement')
            ax[0].grid(True)
            
            # Plot average distance from center over time
            ax[1].plot(results._data.get('avg_distance', []), 'r-')
            ax[1].set_xlabel('Time')
            ax[1].set_ylabel('Average Distance from Center')
            ax[1].set_title('Agent Distance from Center')
            ax[1].grid(True)
            
            plt.tight_layout()
            plt.savefig('bouncing_agents_results.png')
            print("Results plotted and saved to 'bouncing_agents_results.png'.")
        except Exception as e:
            print(f"Error plotting results: {e}")
    else:
        print("No variables found in results.")
    
    # Return results
    return results


# Run simulation if script is executed directly
if __name__ == "__main__":
    run_simulation()