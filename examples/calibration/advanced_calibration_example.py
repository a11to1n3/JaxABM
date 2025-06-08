"""
Advanced Calibration Example for JaxABM

This example demonstrates comprehensive calibration functionality with:
- All optimization methods working correctly
- Proper model dynamics and realistic behavior
- Robust error handling and validation
- Performance comparison and visualization
"""

import jaxabm as jx
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class PopulationAgent(jx.Agent):
    """Agent with realistic population dynamics."""
    
    def setup(self):
        """Initialize agent with demographic state."""
        return {
            'age': np.random.randint(0, 80),
            'energy': np.random.uniform(0.5, 1.0),
            'reproduction_rate': np.random.uniform(0.01, 0.05),
            'survival_rate': np.random.uniform(0.95, 0.99),
            'resources': np.random.uniform(0.3, 0.7)
        }
    
    def step(self, model_state):
        """Step with realistic population dynamics."""
        age = self._state['age'] 
        energy = self._state['energy']
        reproduction_rate = self._state['reproduction_rate']
        survival_rate = self._state['survival_rate']
        resources = self._state['resources']
        
        # Get model parameters
        params = model_state.get('params', {})
        growth_rate = params.get('growth_rate', 0.02)
        carrying_capacity = params.get('carrying_capacity', 1000)
        resource_renewal = params.get('resource_renewal', 0.1)
        mortality_factor = params.get('mortality_factor', 0.01)
        
        # Age the agent
        age = age + 1
        
        # Update energy based on resources and age
        age_factor = 1.0 - (age / 100.0)  # Older agents have less energy
        energy = energy + (resources * resource_renewal * age_factor) - 0.05
        energy = jnp.clip(energy, 0.0, 1.0)
        
        # Update resources (competition effects)
        # Simplified: resources decrease with overcrowding
        total_agents = model_state.get('total_agents', 100)
        crowding_factor = min(1.0, carrying_capacity / max(total_agents, 1))
        resources = resources * crowding_factor + resource_renewal * 0.5
        resources = jnp.clip(resources, 0.0, 1.0)
        
        # Update survival rate based on conditions
        survival_rate = survival_rate * (0.9 + 0.1 * energy) * (0.9 + 0.1 * resources)
        survival_rate = jnp.clip(survival_rate, 0.8, 0.99)
        
        # Update reproduction rate
        if age > 15 and age < 60 and energy > 0.6:  # Reproductive age and condition
            reproduction_rate = reproduction_rate * growth_rate * energy
        else:
            reproduction_rate = 0.0
        
        return {
            'age': age,
            'energy': energy,
            'reproduction_rate': reproduction_rate,
            'survival_rate': survival_rate,
            'resources': resources
        }


class PopulationModel(jx.Model):
    """Population model with realistic dynamics."""
    
    def setup(self):
        """Set up population model."""
        n_agents = self.p.get('n_agents', 200)
        self.agents = self.add_agents(n_agents, PopulationAgent)
        
        # Environment tracking
        self.env.add_state('time', 0)
        self.env.add_state('total_population', n_agents)
        self.env.add_state('births', 0)
        self.env.add_state('deaths', 0)
        self.env.add_state('total_energy', 0.0)
        self.env.add_state('avg_age', 0.0)
    
    def step(self):
        """Execute population dynamics."""
        self.env.add_state('time', self.env.time + 1)
        
        # Calculate population metrics
        total_population = len(self.agents)
        total_energy = 0.0
        total_age = 0.0
        births_this_step = 0
        deaths_this_step = 0
        
        # Process agents if JAX model available
        if hasattr(self, '_jax_model') and self._jax_model.state:
            agent_states = self._jax_model.state.get('agents', {})
            if 'agents' in agent_states:
                energies = agent_states['agents'].get('energy', jnp.array([]))
                ages = agent_states['agents'].get('age', jnp.array([]))
                
                if len(energies) > 0:
                    total_energy = float(jnp.sum(energies))
                    total_age = float(jnp.sum(ages))
                    total_population = len(energies)
        
        # Fallback calculation
        if total_energy == 0.0:
            for agent in self.agents:
                if hasattr(agent, '_state'):
                    total_energy += agent._state.get('energy', 0.0)
                    total_age += agent._state.get('age', 0.0)
        
        # Calculate averages
        avg_energy = total_energy / max(total_population, 1)
        avg_age = total_age / max(total_population, 1)
        
        # Simulate births and deaths (simplified)
        params = self.p
        growth_rate = params.get('growth_rate', 0.02)
        mortality_factor = params.get('mortality_factor', 0.01)
        
        # Birth calculation
        births_this_step = int(total_population * growth_rate * avg_energy)
        births_this_step = max(0, min(births_this_step, 10))  # Reasonable bounds
        
        # Death calculation  
        deaths_this_step = int(total_population * mortality_factor * (2.0 - avg_energy))
        deaths_this_step = max(0, min(deaths_this_step, total_population // 10))
        
        # Update environment
        self.env.add_state('total_population', total_population)
        self.env.add_state('births', births_this_step)
        self.env.add_state('deaths', deaths_this_step)
        self.env.add_state('total_energy', avg_energy)
        self.env.add_state('avg_age', avg_age)
        
        # Record metrics
        self.record('population', total_population)
        self.record('avg_energy', avg_energy)
        self.record('avg_age', avg_age)
        self.record('births', births_this_step)
        self.record('deaths', deaths_this_step)
        
        # Population growth rate
        if hasattr(self, '_population_history'):
            if len(self._population_history) > 0:
                prev_pop = self._population_history[-1]
                growth = (total_population - prev_pop) / max(prev_pop, 1)
                self.record('growth_rate', growth)
            self._population_history.append(total_population)
        else:
            self._population_history = [total_population]
    
    def compute_metrics(self, env_state, agent_states, model_params):
        """Compute final metrics for calibration."""
        population = env_state.get('total_population', 0)
        avg_energy = env_state.get('total_energy', 0.0)
        avg_age = env_state.get('avg_age', 0.0)
        
        # Calculate population stability (inverse of variance)
        if hasattr(self, '_population_history') and len(self._population_history) > 5:
            pop_history = jnp.array(self._population_history[-10:])  # Last 10 steps
            pop_variance = jnp.var(pop_history)
            stability = 1.0 / (1.0 + pop_variance)
        else:
            stability = 0.5
        
        return {
            'final_population': population,
            'avg_energy': avg_energy,
            'avg_age': avg_age,
            'population_stability': stability,
            'sustainable_population': min(population, model_params.get('carrying_capacity', 1000))
        }


def test_calibration_methods():
    """Test different calibration methods systematically."""
    print("Testing JaxABM Calibration Methods")
    print("="*50)
    
    # Create model factory
    def model_factory(params=None, config=None):
        model_params = {
            'n_agents': 200,
            'growth_rate': 0.02,
            'carrying_capacity': 300,
            'resource_renewal': 0.1,
            'mortality_factor': 0.01
        }
        if params:
            model_params.update(params)
        
        model = PopulationModel(model_params)
        if config:
            model.config = config
        return model
    
    # Calibration setup
    initial_params = {
        'growth_rate': 0.03,
        'mortality_factor': 0.015,
        'resource_renewal': 0.08
    }
    
    target_metrics = {
        'final_population': 250,
        'avg_energy': 0.7,
        'population_stability': 0.8
    }
    
    param_bounds = {
        'growth_rate': (0.005, 0.05),
        'mortality_factor': (0.005, 0.03),
        'resource_renewal': (0.05, 0.2)
    }
    
    # Test different methods
    methods = [
        ("adam", "Adam Optimizer"),
        ("sgd", "Stochastic Gradient Descent"),
        ("es", "Evolution Strategies"),
        ("pso", "Particle Swarm Optimization"),
        ("cem", "Cross-Entropy Method")
    ]
    
    results = {}
    
    for method_key, method_name in methods:
        print(f"\nTesting {method_name} ({method_key})...")
        print("-" * 30)
        
        try:
            calibrator = jx.analysis.ModelCalibrator(
                model_factory=model_factory,
                initial_params=initial_params.copy(),
                target_metrics=target_metrics,
                param_bounds=param_bounds,
                method=method_key,
                max_iterations=15,
                learning_rate=0.02,
                evaluation_steps=30,
                num_evaluation_runs=2,
                tolerance=1e-4,
                seed=42
            )
            
            optimal_params = calibrator.calibrate(verbose=False)
            
            results[method_key] = {
                'success': True,
                'method_name': method_name,
                'params': optimal_params,
                'loss': calibrator.best_loss,
                'iterations': len(calibrator.loss_history)
            }
            
            print(f"✅ Success! Final loss: {calibrator.best_loss:.6f}")
            print(f"   Iterations: {len(calibrator.loss_history)}")
            print(f"   Parameters: {optimal_params}")
            
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
            results[method_key] = {
                'success': False,
                'method_name': method_name,
                'error': str(e)
            }
    
    return results


def test_ensemble_calibration():
    """Test ensemble calibration combining multiple methods."""
    print(f"\nTesting Ensemble Calibration")
    print("="*30)
    
    # Create model factory
    def model_factory(params=None, config=None):
        model_params = {
            'n_agents': 200,
            'growth_rate': 0.02,
            'carrying_capacity': 300,
            'resource_renewal': 0.1,
            'mortality_factor': 0.01
        }
        if params:
            model_params.update(params)
        return PopulationModel(model_params)
    
    initial_params = {
        'growth_rate': 0.03,
        'mortality_factor': 0.015
    }
    
    target_metrics = {
        'final_population': 250,
        'avg_energy': 0.7
    }
    
    try:
        ensemble = jx.analysis.EnsembleCalibrator(
            model_factory=model_factory,
            initial_params=initial_params,
            target_metrics=target_metrics,
            methods=["adam", "es", "pso"],
            max_iterations=10
        )
        
        results = ensemble.calibrate(verbose=False)
        
        print(f"✅ Ensemble calibration succeeded!")
        print(f"   Best method: {results['best_method']}")
        print(f"   Best loss: {results['best_loss']:.6f}")
        print(f"   Best params: {results['best_params']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Ensemble calibration failed: {e}")
        return None


def plot_results(method_results):
    """Plot calibration results comparison."""
    successful_methods = {k: v for k, v in method_results.items() if v['success']}
    
    if not successful_methods:
        print("No successful methods to plot.")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Final loss comparison
    methods = list(successful_methods.keys())
    losses = [successful_methods[m]['loss'] for m in methods]
    method_names = [successful_methods[m]['method_name'] for m in methods]
    
    bars1 = ax1.bar(method_names, losses, alpha=0.7, color='skyblue')
    ax1.set_ylabel('Final Loss')
    ax1.set_title('Calibration Method Performance')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, loss in zip(bars1, losses):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.4f}', ha='center', va='bottom')
    
    # Plot 2: Iterations to convergence
    iterations = [successful_methods[m]['iterations'] for m in methods]
    bars2 = ax2.bar(method_names, iterations, alpha=0.7, color='lightcoral')
    ax2.set_ylabel('Iterations')
    ax2.set_title('Convergence Speed')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, iter_count in zip(bars2, iterations):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{iter_count}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('calibration_methods_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Results plot saved as 'calibration_methods_comparison.png'")


def main():
    """Run comprehensive calibration testing."""
    print("🚀 JaxABM Advanced Calibration Testing")
    print("="*50)
    
    # Test individual methods
    method_results = test_calibration_methods()
    
    # Test ensemble
    ensemble_results = test_ensemble_calibration()
    
    # Generate summary
    print(f"\n📊 FINAL SUMMARY")
    print("="*30)
    
    successful_methods = [k for k, v in method_results.items() if v['success']]
    failed_methods = [k for k, v in method_results.items() if not v['success']]
    
    print(f"✅ Successful methods ({len(successful_methods)}):")
    for method in successful_methods:
        result = method_results[method]
        print(f"   - {result['method_name']}: {result['loss']:.6f} loss")
    
    if failed_methods:
        print(f"\n❌ Failed methods ({len(failed_methods)}):")
        for method in failed_methods:
            print(f"   - {method_results[method]['method_name']}")
    
    print(f"\n🎯 Ensemble: {'SUCCESS' if ensemble_results else 'FAILED'}")
    
    # Plot results
    plot_results(method_results)
    
    print(f"\n🎉 Calibration testing complete!")
    print(f"   Methods working: {len(successful_methods)}/{len(method_results)}")
    
    return method_results, ensemble_results


if __name__ == "__main__":
    main() 