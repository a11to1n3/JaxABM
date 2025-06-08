"""
Simple example demonstrating the sensitivity analysis with the AgentPy-like interface.
"""

import jaxabm as jx
import numpy as np
import matplotlib.pyplot as plt


class SimpleGrowthAgent(jx.Agent):
    """Agent with simple growth behavior."""
    
    def setup(self):
        """Initialize agent."""
        return {
            'size': 1.0,
            'growth_rate': 0.0  # Will be set by model
        }
    
    def step(self, model_state):
        """Grow by growth rate."""
        # Get current state
        size = self._state['size']
        
        # Get growth rate from environment
        env_state = model_state.get('env', {})
        growth_rate = env_state.get('growth_rate', 0.1)
        
        # Compute new size
        new_size = size * (1.0 + growth_rate)
        
        # Return updated state
        return {
            'size': new_size,
            'growth_rate': growth_rate
        }


class SimpleModel(jx.Model):
    """Simple model with growing agents."""
    
    def setup(self):
        """Set up model with agents."""
        # Get parameters
        n_agents = self.p.get('n_agents', 10)
        growth_rate = self.p.get('growth_rate', 0.1)
        carrying_capacity = self.p.get('carrying_capacity', 100.0)
        
        # Add agents
        self.agents = self.add_agents(n_agents, SimpleGrowthAgent)
        
        # Environment variables
        self.env.add_state('time', 0)
        self.env.add_state('carrying_capacity', carrying_capacity)
        self.env.add_state('mean_size', 1.0)
        self.env.add_state('growth_rate', growth_rate)  # Store growth rate in environment
    
    def step(self):
        """Update model state."""
        # Update time
        self.env.add_state('time', self.env.time + 1)
        
        # Get agent states
        if hasattr(self, '_jax_model') and self._jax_model.state:
            agent_states = self._jax_model.state.get('agents', {})
            if 'agents' in agent_states and 'size' in agent_states['agents']:
                sizes = agent_states['agents']['size']
                mean_size = float(np.mean(sizes))
                
                # Apply carrying capacity constraint
                carrying_capacity = self.p.get('carrying_capacity', 100.0)
                if mean_size > carrying_capacity:
                    # Record before reset
                    self.record('mean_size_before_reset', mean_size)
                    
                    # Reset sizes
                    if hasattr(self.agents.collection, '_states'):
                        if 'size' in self.agents.collection._states:
                            self.agents.collection._states['size'] = np.ones(len(sizes))
                            mean_size = 1.0
                
                # Update environment state
                self.env.add_state('mean_size', mean_size)
                
                # Record data
                self.record('time', self.env.time)
                self.record('mean_size', mean_size)
    
    def compute_metrics(self, env_state, agent_states, model_params):
        """Compute model metrics."""
        time = env_state.get('time', 0)
        mean_size = env_state.get('mean_size', 1.0)
        carrying_capacity = env_state.get('carrying_capacity', 100.0)
        
        # Calculate resource usage
        resource_usage = mean_size / carrying_capacity
        
        # Calculate efficiency
        growth_rate = model_params.get('growth_rate', 0.1)
        efficiency = mean_size / (time * growth_rate + 1.0)  # +1 to avoid division by zero
        
        return {
            'final_size': mean_size,
            'resource_usage': resource_usage,
            'efficiency': efficiency
        }


def run_sensitivity_analysis():
    """Run sensitivity analysis on the simple model."""
    print("Running sensitivity analysis...")
    
    # Define parameters
    growth_rate = jx.Parameter('growth_rate', bounds=(0.05, 0.3))
    carrying_capacity = jx.Parameter('carrying_capacity', bounds=(50.0, 200.0))
    
    # Create analyzer
    analyzer = jx.SensitivityAnalyzer(
        model_class=SimpleModel,
        parameters=[growth_rate, carrying_capacity],
        n_samples=5,  # Small for example
        metrics=['final_size', 'resource_usage', 'efficiency']
    )
    
    # Run analysis
    results = analyzer.run()
    
    # Calculate sensitivity
    sensitivity = analyzer.calculate_sensitivity()
    
    # Print results
    print("\nSensitivity results:")
    for metric, indices in sensitivity.items():
        print(f"\n{metric}:")
        for param, value in indices.items():
            print(f"  {param}: {value:.4f}")
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    analyzer.plot('final_size', ax=axes[0])
    axes[0].set_title('Sensitivity: Final Size')
    
    analyzer.plot('resource_usage', ax=axes[1])
    axes[1].set_title('Sensitivity: Resource Usage')
    
    analyzer.plot('efficiency', ax=axes[2])
    axes[2].set_title('Sensitivity: Efficiency')
    
    plt.tight_layout()
    plt.savefig('simple_sensitivity_results.png')
    print("\nSensitivity plot saved as 'simple_sensitivity_results.png'")
    
    return sensitivity


if __name__ == "__main__":
    sensitivity = run_sensitivity_analysis()