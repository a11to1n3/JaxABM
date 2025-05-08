"""
Example of using the JAX-based API for a simple economic model.

This example demonstrates how to:
1. Define agent types with JAX
2. Set up a model with multiple agent types
3. Run a model simulation
4. Perform sensitivity analysis
5. Calibrate model parameters
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from typing import Dict, Any
import argparse

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jax import random

# Import components from their correct modules after refactoring
from jaxabm.agent import AgentType, AgentCollection
from jaxabm.core import ModelConfig
from jaxabm.model import Model
from jaxabm.analysis import SensitivityAnalysis, ModelCalibrator


# Define agent types
class Consumer(AgentType):
    """Consumer agent that earns income and purchases goods."""
    
    def __init__(self, income_level=1.0, propensity_to_consume=0.8):
        """Initialize Consumer agent type.
        
        Args:
            income_level: Base income level for all consumers
            propensity_to_consume: Proportion of income spent on consumption
        """
        self.income_level = income_level
        self.propensity_to_consume = propensity_to_consume

    def init_state(self, model_config: ModelConfig, key: jax.Array) -> Dict[str, Any]:
        """Initialize consumer state."""
        # Use own parameters
        return {
            'savings': jnp.array(0.0),
            'consumption': jnp.array(0.0),
            'utility': jnp.array(0.0),
            'income': self.income_level * (0.8 + 0.4 * random.uniform(key))
        }
    
    def update(self, state: Dict[str, Any], model_state: Dict[str, Any], 
               model_config: ModelConfig, key: jax.Array) -> Dict[str, Any]:
        """Update consumer state."""
        # Calculate disposable income (income + market_subsidy from env state)
        disposable_income = state['income'] + model_state['env']['market_subsidy']
        
        # Calculate consumption based on propensity to consume
        consumption = self.propensity_to_consume * disposable_income
        
        # Calculate savings (remaining income)
        savings = state['savings'] + (disposable_income - consumption)
        
        # Calculate utility (logarithmic utility function)
        utility = jnp.log(consumption + 1.0)
        
        # Update state dictionary
        new_state = {
            'savings': savings,
            'consumption': consumption,
            'utility': utility,
            'income': state['income'] # Income assumed constant for simplicity here
        }
        
        # Return only the new state
        return new_state


class Producer(AgentType):
    """Producer agent that produces goods and services."""

    def __init__(self, productivity=1.0, capital_level=10.0, reinvestment_rate=0.3):
        """Initialize Producer agent type.
        
        Args:
            productivity: Production efficiency parameter
            capital_level: Initial capital for producers
            reinvestment_rate: Proportion of profits reinvested
        """
        self.productivity = productivity
        self.capital_level = capital_level
        self.reinvestment_rate = reinvestment_rate

    def init_state(self, model_config: ModelConfig, key: jax.Array) -> Dict[str, Any]:
        """Initialize producer state."""
        # Use own parameters
        return {
            'capital': self.capital_level * (0.8 + 0.4 * random.uniform(key)),
            'production': jnp.array(0.0),
            'profit': jnp.array(0.0)
        }
    
    def update(self, state: Dict[str, Any], model_state: Dict[str, Any], 
               model_config: ModelConfig, key: jax.Array) -> Dict[str, Any]:
        """Update producer state."""
        # Calculate production (Cobb-Douglas production function)
        production = self.productivity * (state['capital'] ** 0.7)
        
        # Calculate revenue (production * price_level from env state)
        revenue = production * model_state['env']['price_level']
        
        # Calculate costs (fixed capital cost + variable cost)
        costs = 0.1 * state['capital'] + 0.05 * production
        
        # Calculate profit
        profit = revenue - costs
        
        # Update capital (reinvest some profit)
        capital = state['capital'] + self.reinvestment_rate * profit
        
        # Update state dictionary
        new_state = {
            'capital': capital,
            'production': production,
            'profit': profit
        }
        
        # Return only the new state
        return new_state


# Define model state update function
def update_model_state(env_state: Dict[str, Any], agent_states: Dict[str, Dict[str, Any]], 
                       model_params: Dict[str, Any], key: jax.Array) -> Dict[str, Any]:
    """Update the model state based on agent states and model parameters."""
    
    # Extract agent states
    consumer_states = agent_states.get('consumers')
    producer_states = agent_states.get('producers')
    
    # Calculate total consumption and production from agent states
    total_consumption = jnp.sum(consumer_states['consumption']) if consumer_states else jnp.array(0.0)
    total_production = jnp.sum(producer_states['production']) if producer_states else jnp.array(0.0)
    
    # Update price level based on supply and demand
    price_diff = total_consumption - total_production
    relative_diff = price_diff / (jnp.maximum(total_consumption, total_production) + 1e-8)
    price_adjustment = jnp.clip(relative_diff, -0.5, 0.5) # Limit extreme adjustments
    
    # Apply price adjustment using model parameter
    price_level = env_state['price_level'] * (1.0 + model_params['price_adjustment_rate'] * price_adjustment)
    price_level = jnp.maximum(price_level, 0.1) # Ensure price level stays positive
    
    # Calculate market subsidy using model parameters
    market_subsidy = model_params['subsidy_rate'] * jnp.maximum(0.0, 
        price_level - model_params['target_price']
    )
    
    # Update GDP
    gdp = total_production * price_level
    
    # Update unemployment rate (simplified proxy)
    unemployment = jnp.maximum(0.0, jnp.minimum(1.0, 
        1.0 - total_production / (total_consumption + 1e-8)
    ))
    
    # Create new environment state dictionary
    new_env_state = {
        'price_level': price_level,
        'market_subsidy': market_subsidy,
        'gdp': gdp,
        'unemployment': unemployment,
        'total_consumption': total_consumption, # Store aggregate for metrics
        'total_production': total_production # Store aggregate for metrics
    }
    
    return new_env_state


# Define metrics function
def compute_metrics(env_state: Dict[str, Any], agent_states: Dict[str, Dict[str, Any]], 
                    model_params: Dict[str, Any]) -> Dict[str, Any]:
    """Compute metrics from model state and agent states."""
    
    # Extract agent states
    consumer_states = agent_states.get('consumers')
    producer_states = agent_states.get('producers')
    
    # Calculate average utility
    avg_utility = jnp.mean(consumer_states['utility']) if consumer_states else jnp.array(0.0)
    
    # Calculate average profit
    avg_profit = jnp.mean(producer_states['profit']) if producer_states else jnp.array(0.0)
    
    # Calculate price stability
    price_stability = 1.0 / (jnp.abs(env_state['price_level'] - model_params['target_price']) + 1e-8)
    
    # Calculate gini coefficient (proxy)
    if consumer_states and 'consumption' in consumer_states:
        cons = consumer_states['consumption']
        inequality = jnp.max(cons) / (jnp.min(cons) + 1e-8) if cons.size > 0 else jnp.array(1.0)
    else:
        inequality = jnp.array(1.0) # Default if no consumers or consumption data
    
    return {
        'avg_utility': avg_utility,
        'avg_profit': avg_profit,
        'price_stability': price_stability,
        'inequality': inequality,
        'unemployment': env_state['unemployment'],
        'gdp': env_state['gdp']
    }


# Function to create a model
def create_economy_model(
    num_consumers=1000, 
    num_producers=20, 
    income_level=1.0,
    propensity_to_consume=0.8,
    productivity=1.0,
    capital_level=10.0,
    price_adjustment_rate=0.1,
    subsidy_rate=0.2,
    target_price=1.0,
    seed=0,
    # Add parameters for analysis tools
    params=None,
    config=None
):
    """Create an economy model with the given parameters.
    
    This function can be called directly with individual parameters or via
    analysis tools that provide a params dictionary.

    Args:
        num_consumers: Number of consumer agents
        num_producers: Number of producer agents
        income_level: Base income level for consumers
        propensity_to_consume: Proportion of income spent on consumption
        productivity: Production efficiency parameter
        capital_level: Initial capital for producers
        price_adjustment_rate: Rate of price level adjustment
        subsidy_rate: Government subsidy rate
        target_price: Target price level
        seed: Random seed
        params: Dictionary of parameters (used by analysis tools)
        config: ModelConfig object (used by analysis tools)
    
    Returns:
        Configured model instance
    """
    # If params is provided (by analysis tools), use those parameters
    if params is not None:
        propensity_to_consume = params.get('propensity_to_consume', propensity_to_consume)
        productivity = params.get('productivity', productivity)
        subsidy_rate = params.get('subsidy_rate', subsidy_rate)
    
    # If config is provided, use it instead of creating a new one
    if config is None:
        config = ModelConfig(
            steps=50,
            collect_interval=1,
            seed=seed
        )
    
    # Create agent type instances with specific parameters for this model run
    consumer_agent_type = Consumer(income_level=income_level, 
                                   propensity_to_consume=propensity_to_consume)
    producer_agent_type = Producer(productivity=productivity, 
                                   capital_level=capital_level)

    # Create agent collections
    consumers = AgentCollection(
        agent_type=consumer_agent_type,
        num_agents=num_consumers
    )
    producers = AgentCollection(
        agent_type=producer_agent_type,
        num_agents=num_producers
    )
    
    # Define initial environment state
    initial_env_state = {
        'price_level': jnp.array(1.0),
        'market_subsidy': jnp.array(0.0),
        'gdp': jnp.array(0.0),
        'unemployment': jnp.array(0.0),
        'total_consumption': jnp.array(0.0),
        'total_production': jnp.array(0.0)
    }
    
    # Define model parameters (distinct from agent parameters)
    model_params = {
        'price_adjustment_rate': jnp.array(price_adjustment_rate),
        'subsidy_rate': jnp.array(subsidy_rate),
        'target_price': jnp.array(target_price)
    }
    
    # Create model instance
    model = Model(
        params=model_params,
        config=config,
        update_state_fn=update_model_state,
        metrics_fn=compute_metrics
    )
    
    # Add agent collections
    model.add_agent_collection('consumers', consumers)
    model.add_agent_collection('producers', producers)
    
    # Add initial environment state
    for name, value in initial_env_state.items():
        model.add_env_state(name, value)

    # Initialization is handled by model.run()
    return model


def main():
    """Run the example."""
    
    print("Running JAX-based agent model example...")
    
    # Check if we should skip calibration (which is slow)
    parser = argparse.ArgumentParser(description='Run JAX ABM examples')
    parser.add_argument('--skip-calibration', action='store_true', help='Skip calibration steps (faster)')
    parser.add_argument('--fast', action='store_true', help='Run with minimal agents and steps (much faster)')
    args = parser.parse_args()
    
    # Use fewer agents and steps if fast mode is enabled
    num_agents = 50 if args.fast else 1000
    num_steps = 20 if args.fast else 50
    
    # 1. Basic model run
    print("\n1. Running basic model simulation...")
    model = create_economy_model(num_consumers=num_agents, num_producers=num_agents//50, seed=42)
    results = model.run(steps=num_steps)
    
    # Plot some results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(results['step'], results['gdp'])
    plt.title('GDP over time')
    plt.xlabel('Step')
    plt.ylabel('GDP')
    
    plt.subplot(2, 2, 2)
    plt.plot(results['step'], results['unemployment'])
    plt.title('Unemployment over time')
    plt.xlabel('Step')
    plt.ylabel('Unemployment rate')
    
    plt.subplot(2, 2, 3)
    plt.plot(results['step'], results['avg_utility'])
    plt.title('Consumer Utility over time')
    plt.xlabel('Step')
    plt.ylabel('Average Utility')
    
    plt.subplot(2, 2, 4)
    plt.plot(results['step'], results['inequality'])
    plt.title('Inequality over time')
    plt.xlabel('Step')
    plt.ylabel('Inequality (max/min ratio)')
    
    plt.tight_layout()
    plt.savefig('jax_abm_basic_results.png')
    plt.close()
    
    # 2. Sensitivity analysis
    print("\n2. Running sensitivity analysis...")
    # Use even fewer samples in fast mode
    sa_samples = 3 if args.fast else 10
    
    sensitivity = SensitivityAnalysis(
        model_factory=create_economy_model,
        param_ranges={
            'propensity_to_consume': (0.6, 0.9),
            'productivity': (0.5, 1.5),
            'subsidy_rate': (0.0, 0.4)
        },
        metrics_of_interest=['gdp', 'unemployment', 'inequality'],
        num_samples=sa_samples,
        seed=42
    )
    
    # Run the analysis
    sensitivity_results = sensitivity.run()
    
    # Calculate Sobol indices
    sobol_indices = sensitivity.sobol_indices()
    
    # Print results
    print("\nSensitivity analysis results (Sobol indices):")
    for metric, indices in sobol_indices.items():
        print(f"  {metric}:")
        for param, index in indices.items():
            print(f"    {param}: {index:.4f}")
    
    # Plot sensitivity results
    plt.figure(figsize=(10, 6))
    
    metrics = list(sobol_indices.keys())
    params = list(sobol_indices[metrics[0]].keys())
    
    x = np.arange(len(metrics))
    width = 0.2
    multiplier = 0
    
    for param in params:
        param_values = [sobol_indices[metric][param] for metric in metrics]
        offset = width * multiplier
        plt.bar(x + offset, param_values, width, label=param)
        multiplier += 1
    
    plt.xlabel('Metrics')
    plt.ylabel('Sensitivity (Sobol index)')
    plt.title('Parameter Sensitivity Analysis')
    plt.xticks(x + width, metrics)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('jax_abm_sensitivity.png')
    plt.close()
    
    # 3. Model calibration
    if args.skip_calibration:
        print("\n3. Skipping model calibration (use --skip-calibration=False to run)")
        # Use dummy values for the plots
        optimal_params_gd = {'propensity_to_consume': 0.85, 'productivity': 1.2, 'subsidy_rate': 0.15}
        optimal_params_rl = {'propensity_to_consume': 0.82, 'productivity': 1.1, 'subsidy_rate': 0.12}
    else:
        print("\n3. Running model calibration (this may take several minutes)...")
        
        # Define target metrics
        target_metrics = {
            'gdp': 130.0,
            'unemployment': 0.05,
            'inequality': 3.0
        }
        
        # Create a simplified model factory for calibration (fewer agents and steps)
        def calibration_model_factory(**kwargs):
            # Use a very reduced model for calibration to speed things up
            model = create_economy_model(
                num_consumers=20,  # Much fewer agents for faster gradient computation
                num_producers=2,   # Much fewer agents
                seed=42,
                **kwargs
            )
            # Return the model object itself, not the result of running it
            return model
        
        # Initialize calibrator with gradient descent
        calibrator_gd = ModelCalibrator(
            model_factory=calibration_model_factory,  # Use the simplified factory
            initial_params={
                'propensity_to_consume': 0.7,
                'productivity': 1.0,
                'subsidy_rate': 0.1
            },
            target_metrics=target_metrics,
            metrics_weights={
                'gdp': 0.01,  # Scale down because GDP has larger values
                'unemployment': 1.0,
                'inequality': 0.1
            },
            learning_rate=0.05,
            max_iterations=3,  # Reduced iterations
            method='gradient'
        )
        
        # Run calibration
        optimal_params_gd = calibrator_gd.calibrate()
        
        print("\nCalibration results (Gradient Descent):")
        print(f"  Optimal parameters: {optimal_params_gd}")
        print(f"  Final loss: {calibrator_gd.loss_history[-1]}")
        
        # Reuse the same factory for RL to be consistent
        # Initialize calibrator with RL approach
        calibrator_rl = ModelCalibrator(
            model_factory=calibration_model_factory,  # Use the simplified factory
            initial_params={
                'propensity_to_consume': 0.7,
                'productivity': 1.0,
                'subsidy_rate': 0.1
            },
            target_metrics=target_metrics,
            metrics_weights={
                'gdp': 0.01,
                'unemployment': 1.0,
                'inequality': 0.1
            },
            learning_rate=0.05,
            max_iterations=3,  # Reduced iterations
            method='rl'
        )
        
        # Run calibration
        optimal_params_rl = calibrator_rl.calibrate()
        
        print("\nCalibration results (RL):")
        print(f"  Optimal parameters: {optimal_params_rl}")
        print(f"  Final loss: {calibrator_rl.loss_history[-1]}")
    
    # 4. Compare model runs with original vs calibrated parameters
    print("\n4. Comparing model runs with original vs calibrated parameters...")
    
    # Original model
    model_original = create_economy_model(num_consumers=num_agents, num_producers=num_agents//50, seed=42)
    results_original = model_original.run(steps=num_steps)
    
    # Calibrated model (using GD results)
    model_calibrated = create_economy_model(
        num_consumers=num_agents,
        num_producers=num_agents//50,
        propensity_to_consume=optimal_params_gd['propensity_to_consume'],
        productivity=optimal_params_gd['productivity'],
        subsidy_rate=optimal_params_gd['subsidy_rate'],
        seed=42
    )
    results_calibrated = model_calibrated.run(steps=num_steps)
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    metrics_to_plot = ['gdp', 'unemployment', 'inequality']
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(len(metrics_to_plot), 1, i + 1)
        plt.plot(results_original['step'], results_original[metric], label='Original')
        plt.plot(results_calibrated['step'], results_calibrated[metric], label='Calibrated')
        # Only show target line in non-skipped calibration mode
        if not args.skip_calibration and metric in target_metrics:
            plt.axhline(y=target_metrics[metric], color='r', linestyle='--', label='Target')
        plt.title(f'{metric.capitalize()} over time')
        plt.xlabel('Step')
        plt.ylabel(metric.capitalize())
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('jax_abm_comparison.png')
    plt.close()
    
    print("\nExample completed. Check the generated PNG files for visualization.")
    print("\nTo run a faster version: python examples/jax_abm_example.py --fast")
    print("To skip calibration: python examples/jax_abm_example.py --skip-calibration")


if __name__ == "__main__":
    main() 