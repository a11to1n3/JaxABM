"""
Example demonstrating custom agent methods with the AgentPy-like interface.

This example shows how agents can have custom methods that can be called
outside of the step function during simulation.
"""

import jaxabm as jx
import numpy as np
import matplotlib.pyplot as plt


class CustomAgent(jx.Agent):
    """Agent with custom methods beyond setup and step."""
    
    def setup(self):
        """Initialize agent state."""
        return {
            'x': 0.5,
            'y': 0.5,
            'energy': 100.0,
            'color_index': 0,  # 0=blue, 1=red, 2=green, etc. (using numeric values for JAX compatibility)
            'custom_actions': 0
        }
    
    def step(self, model_state):
        """Standard update during simulation steps."""
        # Get current state
        x = self._state['x']
        y = self._state['y']
        energy = self._state['energy']
        
        # Move agent (simple deterministic movement)
        x = (x + 0.01) % 1.0
        y = (y + 0.01) % 1.0
        
        # Decrease energy
        energy = max(0.0, energy - 1.0)
        
        # Return updated state
        return {
            'x': x,
            'y': y,
            'energy': energy,
            'color': self._state['color'],
            'custom_actions': self._state['custom_actions']
        }
    
    def change_color(self, new_color):
        """Custom method to change agent color.
        
        This can be called outside of the step function.
        """
        # Update agent's color
        self.color = new_color  # This calls update_state via __setattr__
        
        # Increment custom action counter
        new_state = {
            'custom_actions': self._state['custom_actions'] + 1
        }
        self.update_state(new_state)
        
        return self.color
    
    def boost_energy(self, amount=10.0):
        """Custom method to give the agent an energy boost.
        
        This can be called outside of the step function.
        """
        # Get current energy
        current_energy = self._state['energy']
        
        # Increase energy
        new_energy = current_energy + amount
        
        # Update state
        self.energy = new_energy  # This calls update_state via __setattr__
        
        # Increment custom action counter
        new_state = {
            'custom_actions': self._state['custom_actions'] + 1
        }
        self.update_state(new_state)
        
        return new_energy


class CustomModel(jx.Model):
    """Model demonstrating custom agent methods."""
    
    def setup(self):
        """Set up model with agents and environment."""
        # Add agents
        self.agents = self.add_agents(10, CustomAgent)
        
        # Environment variables
        self.env.add_state('time', 0)
        self.env.add_state('total_energy', 1000.0)
        self.env.add_state('boost_frequency', 10)  # How often to boost agent energy
        self.env.add_state('color_change_frequency', 20)  # How often to change agent colors
    
    def step(self):
        """Execute model logic each step."""
        # Update time
        current_time = self.env.time
        self.env.add_state('time', current_time + 1)
        
        # Every boost_frequency steps, boost a random agent's energy
        if current_time > 0 and current_time % self.env.boost_frequency == 0:
            # Choose a random agent to boost
            agent_id = np.random.randint(len(self.agents))
            
            # Get the agent instance
            agent = self.get_agent('agents', agent_id)
            
            if agent:
                # Call custom method
                new_energy = agent.boost_energy(20.0)
                print(f"Step {current_time}: Boosted agent {agent_id}'s energy to {new_energy}")
        
        # Every color_change_frequency steps, change all agents' colors
        if current_time > 0 and current_time % self.env.color_change_frequency == 0:
            colors = ['red', 'green', 'blue', 'yellow', 'purple']
            new_color = colors[current_time % len(colors)]
            
            # Change all agents' colors
            for i in range(len(self.agents)):
                agent = self.get_agent('agents', i)
                if agent:
                    # Call custom method
                    agent.change_color(new_color)
            
            print(f"Step {current_time}: Changed all agents' colors to {new_color}")
        
        # Calculate total energy and record it
        if hasattr(self, '_jax_model') and self._jax_model.state:
            agent_states = self._jax_model.state.get('agents', {})
            if 'agents' in agent_states and 'energy' in agent_states['agents']:
                total_energy = float(np.sum(agent_states['agents']['energy']))
                self.env.add_state('total_energy', total_energy)
                self.record('total_energy', total_energy)
        
        # Record time
        self.record('time', current_time)
    
    def compute_metrics(self, env_state, agent_states, model_params):
        """Compute model metrics."""
        time = env_state.get('time', 0)
        total_energy = env_state.get('total_energy', 0.0)
        
        # Calculate average energy per agent
        agent_count = len(self.agents)
        avg_energy = total_energy / max(1, agent_count)
        
        # Count custom actions
        custom_actions = 0
        if 'agents' in agent_states and 'custom_actions' in agent_states['agents']:
            custom_actions = int(np.sum(agent_states['agents']['custom_actions']))
        
        return {
            'time': time,
            'total_energy': total_energy,
            'avg_energy': avg_energy,
            'custom_actions': custom_actions
        }


def run_custom_agent_example():
    """Run the custom agent example."""
    print("Running custom agent example...")
    
    # Create parameters
    parameters = {
        'steps': 50,
        'seed': 42
    }
    
    # Create and run model
    model = CustomModel(parameters)
    results = model.run()
    
    # Print results
    print("\nSimulation results:")
    print(f"Final time: {results._data['time'][-1]}")
    print(f"Final total energy: {results._data['total_energy'][-1]:.2f}")
    print(f"Total custom actions: {results._data['custom_actions'][-1]}")
    
    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot total energy over time
    axes[0].plot(results._data['time'], results._data['total_energy'], 'b-')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Total Energy')
    axes[0].set_title('Total Agent Energy Over Time')
    axes[0].grid(True)
    
    # Plot custom actions over time
    if 'custom_actions' in results._data:
        axes[1].plot(results._data['time'], results._data['custom_actions'], 'r-')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Custom Actions')
        axes[1].set_title('Cumulative Custom Actions Over Time')
        axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('custom_agent_results.png')
    print("\nResults plotted and saved to 'custom_agent_results.png'")
    
    return results


if __name__ == "__main__":
    run_custom_agent_example()