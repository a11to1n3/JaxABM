"""
Example demonstrating sensitivity analysis and model calibration with the 
AgentPy-like interface.

This example uses a simple predator-prey model to show how to use the
sensitivity analysis and model calibration tools.
"""

import jaxabm as jx
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


class Prey(jx.Agent):
    """Prey agent that reproduces and can be eaten by predators."""
    
    def setup(self):
        """Initialize prey agent."""
        return {
            'alive': True,
            'age': 0,
            'energy': 1.0,
            'position': jnp.array([0.0, 0.0])
        }
    
    def step(self, model_state):
        """Update prey agent state."""
        # Get current state
        alive = self._state['alive']
        age = self._state['age']
        energy = self._state['energy']
        position = self._state['position']
        
        # Get model parameters
        params = model_state.get('params', {})
        prey_energy_gain = params.get('prey_energy_gain', 0.1)
        prey_reproduce_threshold = params.get('prey_reproduce_threshold', 1.5)
        
        # Gain energy (simplified model)
        energy = energy + prey_energy_gain
        
        # Age
        age = age + 1
        
        # In a more complete model, we would:
        # - Move
        # - Check for predators
        # - Reproduce if enough energy
        # - Die if too old or no energy
        
        # Return updated state
        return {
            'alive': alive,
            'age': age,
            'energy': energy,
            'position': position
        }


class Predator(jx.Agent):
    """Predator agent that hunts prey."""
    
    def setup(self):
        """Initialize predator agent."""
        return {
            'alive': True,
            'age': 0,
            'energy': 1.0,
            'position': jnp.array([0.0, 0.0])
        }
    
    def step(self, model_state):
        """Update predator agent state."""
        # Get current state
        alive = self._state['alive']
        age = self._state['age']
        energy = self._state['energy']
        position = self._state['position']
        
        # Get model parameters
        params = model_state.get('params', {})
        predator_energy_loss = params.get('predator_energy_loss', 0.1)
        predator_energy_gain = params.get('predator_energy_gain', 0.5)
        
        # Lose energy
        energy = energy - predator_energy_loss
        
        # In a more complete model, we would:
        # - Move
        # - Hunt prey
        # - Reproduce if enough energy
        # - Die if too old or no energy
        
        # Return updated state
        return {
            'alive': alive,
            'age': age,
            'energy': energy,
            'position': position
        }


class PredatorPreyModel(jx.Model):
    """Simple predator-prey model."""
    
    def setup(self):
        """Set up model with agents and environment."""
        # Add prey agents
        n_prey = self.p.get('n_prey', 100)
        self.prey = self.add_agents(n_prey, Prey)
        
        # Add predator agents
        n_predators = self.p.get('n_predators', 20)
        self.predators = self.add_agents(n_predators, Predator)
        
        # Environment variables
        self.env.add_state('time', 0)
        self.env.add_state('prey_count', n_prey)
        self.env.add_state('predator_count', n_predators)
    
    def step(self):
        """Execute model logic each step."""
        # Update environment variables
        self.env.add_state('time', self.env.time + 1)
        
        # Simulate basic population dynamics
        params = self.p
        prey_birth_rate = params.get('prey_energy_gain', 0.1) * 0.5
        predator_death_rate = params.get('predator_energy_loss', 0.1) * 0.8
        
        # Calculate population changes (simplified dynamics)
        current_prey = self.env.prey_count
        current_predators = self.env.predator_count
        
        # Prey population change (births and deaths from predation)
        prey_births = jnp.round(current_prey * prey_birth_rate * 0.1).astype(int)
        prey_deaths = jnp.round(current_prey * current_predators * 0.001).astype(int)  # predation
        new_prey = jnp.maximum(1, current_prey + prey_births - prey_deaths)
        
        # Predator population change 
        predator_births = jnp.round(current_predators * prey_deaths * 0.1).astype(int)  # from successful hunting
        predator_deaths = jnp.round(current_predators * predator_death_rate * 0.2).astype(int)
        new_predators = jnp.maximum(1, current_predators + predator_births - predator_deaths)
        
        # Update counts with some bounds
        self.env.add_state('prey_count', jnp.minimum(new_prey, 500))
        self.env.add_state('predator_count', jnp.minimum(new_predators, 100))
        
        # Get agent states for energy calculations
        mean_prey_energy = 0.5 + params.get('prey_energy_gain', 0.1)
        mean_predator_energy = 0.5 + params.get('predator_energy_gain', 0.5) - params.get('predator_energy_loss', 0.1)
        
        # Record metrics
        self.record('prey_count', self.env.prey_count)
        self.record('predator_count', self.env.predator_count)
        self.record('mean_prey_energy', mean_prey_energy)
        self.record('mean_predator_energy', mean_predator_energy)
    
    def compute_metrics(self, env_state, agent_states, model_params):
        """Compute model metrics."""
        # Get counts
        prey_count = env_state.get('prey_count', 0)
        predator_count = env_state.get('predator_count', 0)
        
        # Get prey energy
        prey_states = agent_states.get('prey', {})
        prey_energy = prey_states.get('energy', jnp.array([]))
        mean_prey_energy = jnp.mean(prey_energy) if len(prey_energy) > 0 else jnp.array(0)
        
        # Get predator energy
        predator_states = agent_states.get('predators', {})
        predator_energy = predator_states.get('energy', jnp.array([]))
        mean_predator_energy = jnp.mean(predator_energy) if len(predator_energy) > 0 else jnp.array(0)
        
        # Calculate ratio (avoid division by zero)
        prey_predator_ratio = (prey_count / jnp.maximum(predator_count, 1))
        
        # Return metrics
        return {
            'prey_count': prey_count,
            'predator_count': predator_count,
            'mean_prey_energy': mean_prey_energy,
            'mean_predator_energy': mean_predator_energy,
            'prey_predator_ratio': prey_predator_ratio
        }


def run_sensitivity_analysis():
    """Run sensitivity analysis on the predator-prey model."""
    print("Running sensitivity analysis...")
    
    # Define parameters for sensitivity analysis
    prey_energy_gain = jx.Parameter(
        name='prey_energy_gain',
        bounds=(0.05, 0.2)
    )
    
    predator_energy_loss = jx.Parameter(
        name='predator_energy_loss',
        bounds=(0.05, 0.2)
    )
    
    predator_energy_gain = jx.Parameter(
        name='predator_energy_gain',
        bounds=(0.3, 0.7)
    )
    
    # Create sensitivity analyzer
    analyzer = jx.SensitivityAnalyzer(
        model_class=PredatorPreyModel,
        parameters=[prey_energy_gain, predator_energy_loss, predator_energy_gain],
        n_samples=5,  # Small number for quick example
        metrics=['prey_count', 'predator_count', 'prey_predator_ratio']
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
    
    # Plot sensitivity
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    analyzer.plot('prey_count', ax=axes[0])
    axes[0].set_title('Sensitivity: Prey Count')
    
    analyzer.plot('predator_count', ax=axes[1])
    axes[1].set_title('Sensitivity: Predator Count')
    
    analyzer.plot('prey_predator_ratio', ax=axes[2])
    axes[2].set_title('Sensitivity: Prey/Predator Ratio')
    
    plt.tight_layout()
    plt.savefig('sensitivity_results.png')
    print("\nSensitivity plot saved as 'sensitivity_results.png'")
    
    return results, sensitivity


def run_model_calibration():
    """Run model calibration on the predator-prey model."""
    print("Running model calibration...")
    
    # Define parameters for calibration
    prey_energy_gain = jx.Parameter(
        name='prey_energy_gain',
        bounds=(0.05, 0.2)
    )
    
    predator_energy_loss = jx.Parameter(
        name='predator_energy_loss',
        bounds=(0.05, 0.2)
    )
    
    # Define target metrics
    target_metrics = {
        'prey_count': 150,
        'predator_count': 30,
        'prey_predator_ratio': 5.0
    }
    
    # Create calibrator
    calibrator = jx.ModelCalibrator(
        model_class=PredatorPreyModel,
        parameters=[prey_energy_gain, predator_energy_loss],
        target_metrics=target_metrics,
        metrics_weights={
            'prey_count': 1.0,
            'predator_count': 1.0,
            'prey_predator_ratio': 2.0
        },
        learning_rate=0.01,
        max_iterations=5,  # Small number for quick example
        method='es'  # Use Evolution Strategies (working method)
    )
    
    # Run calibration
    optimal_params = calibrator.run()
    
    # Print results
    print("\nCalibration results:")
    for param, value in optimal_params.items():
        print(f"  {param}: {value:.4f}")
    
    # Plot calibration progress
    plt.figure(figsize=(8, 5))
    calibrator.plot_progress()
    plt.tight_layout()
    plt.savefig('calibration_results.png')
    print("\nCalibration plot saved as 'calibration_results.png'")
    
    return optimal_params


if __name__ == "__main__":
    # Run sensitivity analysis
    sensitivity_results, sensitivity = run_sensitivity_analysis()
    
    # Run model calibration
    optimal_params = run_model_calibration()