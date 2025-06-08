"""
JaxABM Basic Example - Simple working demonstration

This example demonstrates the core features of JaxABM:
- Basic agent definition and behavior
- Model setup and execution
- Results analysis and visualization

This is the best starting point for learning JaxABM.
"""

import jaxabm as jx
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time


class RandomWalker(jx.Agent):
    """Agent that performs a random walk with bounds."""
    
    def setup(self):
        """Initialize agent state."""
        # Start at a random position between 0 and 1
        return {
            'position': jnp.array([0.5, 0.5]),
            'velocity': jnp.array([0.01, 0.01]),
            'color': 0,  # 0 = red, 1 = blue
            'steps_taken': 0
        }
    
    def step(self, model_state):
        """Update agent state."""
        # Get current state
        position = self._state['position']
        velocity = self._state['velocity']
        color = self._state['color']
        steps_taken = self._state['steps_taken'] + 1
        
        # Get bounds from environment
        env_state = model_state.get('env', {})
        bounds = env_state.get('bounds', jnp.array([0.0, 1.0]))
        
        # Get new position
        new_position = position + velocity
        
        # Bounce off walls
        # When we hit a boundary, reverse velocity in that dimension
        x_bounce = (new_position[0] <= bounds[0]) | (new_position[0] >= bounds[1])
        y_bounce = (new_position[1] <= bounds[0]) | (new_position[1] >= bounds[1])
        
        # Update velocity if bouncing
        new_velocity = velocity * jnp.array([1 - 2 * x_bounce, 1 - 2 * y_bounce])
        
        # Update position (ensure within bounds)
        new_position = jnp.clip(new_position, bounds[0], bounds[1])
        
        # Change color if bouncing (JAX-compatible way)
        any_bounce = jnp.logical_or(x_bounce, y_bounce)
        new_color = jnp.where(any_bounce, 1 - color, color)  # Toggle between 0 and 1
        
        # Return updated state
        return {
            'position': new_position,
            'velocity': new_velocity,
            'color': new_color,
            'steps_taken': steps_taken
        }


class RandomWalkModel(jx.Model):
    """Model with random walking agents."""
    
    def setup(self):
        """Set up model with agents and environment."""
        # Add agents
        n_agents = self.p.get('n_agents', 50)
        self.walkers = self.add_agents(n_agents, RandomWalker)
        
        # Add environment variables
        self.env.add_state('bounds', jnp.array([0.0, 1.0]))
        self.env.add_state('time', 0)
        
        # Initialize metrics
        self.env.add_state('mean_x', 0.5)
        self.env.add_state('mean_y', 0.5)
        self.env.add_state('num_red', n_agents)
        self.env.add_state('num_blue', 0)
    
    def step(self):
        """Execute model logic each step."""
        # Update environment time
        # Note: Agents are updated automatically by the framework
        if hasattr(self, '_jax_model') and self._jax_model.state and 'env' in self._jax_model.state:
            time = self._jax_model.state['env'].get('time', 0)
            self._jax_model.add_env_state('time', time + 1)
            
            # Record data after agent updates
            self.record_data()
    
    def record_data(self):
        """Record data for analysis."""
        # Only record if the JAX model is initialized
        if not hasattr(self, '_jax_model') or not self._jax_model.state:
            return
        
        # Get agent states
        agent_states = self._jax_model.state.get('agents', {})
        if 'walkers' not in agent_states:
            return
        
        walker_states = agent_states['walkers']
        if 'position' not in walker_states or 'color' not in walker_states:
            return
        
        positions = walker_states['position']
        colors = walker_states['color']
        
        # Calculate metrics
        if len(positions) > 0:
            mean_pos = jnp.mean(positions, axis=0)
            mean_x, mean_y = mean_pos[0], mean_pos[1]
            
            # Count agents by color
            num_red = jnp.sum(colors == 0)
            num_blue = jnp.sum(colors == 1)
            
            # Record metrics to environment state
            self._jax_model.add_env_state('mean_x', mean_x)
            self._jax_model.add_env_state('mean_y', mean_y)
            self._jax_model.add_env_state('num_red', num_red)
            self._jax_model.add_env_state('num_blue', num_blue)
            
            # Record metrics for results
            self.record('mean_x', float(mean_x))
            self.record('mean_y', float(mean_y))
            self.record('num_red', int(num_red))
            self.record('num_blue', int(num_blue))
    
    def compute_metrics(self, env_state, agent_states, model_params):
        """Compute model metrics."""
        # Get the necessary data
        time = env_state.get('time', 0)
        mean_x = env_state.get('mean_x', 0.5)
        mean_y = env_state.get('mean_y', 0.5)
        num_red = env_state.get('num_red', 0)
        num_blue = env_state.get('num_blue', 0)
        
        # Get agent positions
        if 'walkers' in agent_states:
            walker_states = agent_states['walkers']
            if 'position' in walker_states:
                positions = walker_states['position']
                if len(positions) > 0:
                    # Calculate distance from center
                    center = jnp.array([0.5, 0.5])
                    distances = jnp.sqrt(jnp.sum((positions - center) ** 2, axis=1))
                    mean_distance = jnp.mean(distances)
                    max_distance = jnp.max(distances)
                    
                    # Return metrics
                    return {
                        'mean_x': mean_x,
                        'mean_y': mean_y,
                        'mean_distance': mean_distance,
                        'max_distance': max_distance,
                        'num_red': num_red,
                        'num_blue': num_blue,
                        'time': time
                    }
        
        # Default metrics if agent data not available
        return {
            'mean_x': mean_x,
            'mean_y': mean_y,
            'mean_distance': 0.0,
            'max_distance': 0.0,
            'num_red': num_red,
            'num_blue': num_blue,
            'time': time
        }


def run_basic_example():
    """Run random walk model and visualize results."""
    # Define parameters
    parameters = {
        'n_agents': 50,
        'steps': 100,
        'seed': 42
    }
    
    # Create and run model
    print("Creating model...")
    model = RandomWalkModel(parameters)
    
    print("Running simulation...")
    start_time = time.time()
    results = model.run()
    end_time = time.time()
    
    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot mean position over time
    if 'mean_x' in results._data and 'mean_y' in results._data:
        axes[0, 0].plot(results._data['mean_x'], results._data['mean_y'], 'b-')
        axes[0, 0].set_xlabel('Mean X Position')
        axes[0, 0].set_ylabel('Mean Y Position')
        axes[0, 0].set_title('Mean Agent Position')
        axes[0, 0].grid(True)
        axes[0, 0].set_xlim(0, 1)
        axes[0, 0].set_ylim(0, 1)
    
    # Plot mean distance from center over time
    if 'mean_distance' in results._data:
        axes[0, 1].plot(results._data['mean_distance'], 'r-')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Mean Distance from Center')
        axes[0, 1].set_title('Agent Distance from Center')
        axes[0, 1].grid(True)
    
    # Plot number of red and blue agents over time
    if 'num_red' in results._data and 'num_blue' in results._data:
        axes[1, 0].plot(results._data['num_red'], 'r-', label='Red')
        axes[1, 0].plot(results._data['num_blue'], 'b-', label='Blue')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Number of Agents')
        axes[1, 0].set_title('Agents by Color')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Plot max distance from center over time
    if 'max_distance' in results._data:
        axes[1, 1].plot(results._data['max_distance'], 'g-')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Maximum Distance from Center')
        axes[1, 1].set_title('Maximum Agent Distance')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('random_walk_results.png')
    print("Results plotted and saved to 'random_walk_results.png'.")
    
    return results


if __name__ == "__main__":
    # Run the example
    results = run_basic_example()
    
    print("\n🎉 Example completed successfully!")
    print("\n💡 Next Steps:")
    print("   • Try modifying agent behavior in RandomWalker.step()")
    print("   • Experiment with different parameter values")
    print("   • Check out examples/calibration/ for parameter optimization")
    print("   • Explore examples/models/ for more complex models")