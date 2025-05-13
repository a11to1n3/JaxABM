"""
Very simple model to test the basic functionality of the AgentPy-like interface.
"""

import jaxabm as jx
import jax.numpy as jnp
import matplotlib.pyplot as plt


class SimpleAgent(jx.Agent):
    """A very simple agent that just increments its state."""
    
    def setup(self):
        """Initialize with a counter and position."""
        return {
            'counter': 0,
            'x': 0.5,
            'y': 0.5
        }
    
    def step(self, model_state):
        """Just increment the counter and move the position."""
        return {
            'counter': self._state['counter'] + 1,
            'x': (self._state['x'] + 0.01) % 1.0,
            'y': (self._state['y'] + 0.01) % 1.0
        }


class SimpleModel(jx.Model):
    """Simple model with a counter and agents."""
    
    def setup(self):
        """Set up the model with agents and a time counter."""
        # Add agents
        self.agents = self.add_agents(10, SimpleAgent)
        
        # Set up environment
        self.env.add_state('time', 0)
    
    def step(self):
        """Just increment the time counter."""
        # Get current time from environment
        time = self.env.time
        
        # Update time
        self.env.add_state('time', time + 1)
        
        # Record metrics
        self.record('time', time)
        self.record('mean_counter', jnp.mean(self.agents.counter))


def run_simple_model():
    """Run the simple model and plot results."""
    # Create and run model
    model = SimpleModel({'steps': 10})
    results = model.run()
    
    # Print results
    print("Results:", results._data)
    
    # Plot results
    if 'mean_counter' in results._data:
        plt.figure(figsize=(8, 5))
        plt.plot(results._data['mean_counter'], 'b-o')
        plt.xlabel('Step')
        plt.ylabel('Mean Counter')
        plt.title('Agent Counter Progress')
        plt.grid(True)
        plt.savefig('simple_model_results.png')
        plt.close()
        print("Results plotted to 'simple_model_results.png'")
    
    return results


if __name__ == "__main__":
    run_simple_model()