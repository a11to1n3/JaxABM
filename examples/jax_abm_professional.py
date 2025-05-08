"""
Professional demonstration of the JAX-accelerated agent-based modeling framework.

This example shows how to use the new API to:
1. Define agent types with clear separation of concerns
2. Create and run models with multiple agent types
3. Perform sensitivity analysis
4. Calibrate model parameters
"""

import os
import sys
import argparse
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jax import random

from jaxabm.agent import AgentType, AgentCollection
from jaxabm.model import Model
from jaxabm.analysis import SensitivityAnalysis, ModelCalibrator
from jaxabm.utils import convert_to_numpy
from jaxabm.core import ModelConfig


# Define agent types as separate classes
class Consumer(AgentType):
    """Consumer agent that earns income and purchases goods.
    
    Consumers earn income, receive subsidies, and make consumption decisions
    based on their propensity to consume.
    """
    
    def __init__(self, income_level=1.0, propensity_to_consume=0.8):
        """Initialize Consumer agent type.
        
        Args:
            income_level: Base income level for all consumers
            propensity_to_consume: Proportion of income spent on consumption
        """
        self.income_level = income_level
        self.propensity_to_consume = propensity_to_consume

    def init_state(self, model_config: ModelConfig, key: jax.Array) -> Dict[str, Any]:
        """Initialize consumer state.
        
        Args:
            model_config: Model configuration (contains seed)
            key: JAX random key
            
        Returns:
            Initial agent state dictionary
        """
        return {
            'savings': jnp.array(0.0),
            'consumption': jnp.array(0.0),
            'utility': jnp.array(0.0),
            'income': self.income_level * (0.8 + 0.4 * random.uniform(key))
        }
    
    def update(self, state: Dict[str, Any], model_state: Dict[str, Any], 
               model_config: ModelConfig, key: jax.Array) -> Dict[str, Any]:
        """Update consumer state.
        
        Args:
            state: Current agent state
            model_state: Current model state (env + other agents)
            model_config: Model configuration
            key: JAX random key
            
        Returns:
            Updated agent state dictionary
        """
        # Calculate disposable income (income + market_subsidy from model env state)
        disposable_income = state['income'] + model_state['env']['market_subsidy']
        
        # Calculate consumption based on propensity to consume
        consumption = self.propensity_to_consume * disposable_income
        
        # Calculate savings (remaining income)
        savings = state['savings'] + (disposable_income - consumption)
        
        # Calculate utility (logarithmic utility function)
        utility = jnp.log(consumption + 1.0)
        
        # Return updated state
        return {
            'savings': savings,
            'consumption': consumption,
            'utility': utility,
            'income': state['income']
        }


class Producer(AgentType):
    """Producer agent that produces goods and services.
    
    Producers transform capital into production using a Cobb-Douglas production
    function, generate revenue based on market prices, and reinvest profits.
    """
    
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
        """Initialize producer state.
        
        Args:
            model_config: Model configuration
            key: JAX random key
            
        Returns:
            Initial agent state dictionary
        """
        return {
            'capital': self.capital_level * (0.8 + 0.4 * random.uniform(key)),
            'production': jnp.array(0.0),
            'profit': jnp.array(0.0)
        }
    
    def update(self, state: Dict[str, Any], model_state: Dict[str, Any], 
               model_config: ModelConfig, key: jax.Array) -> Dict[str, Any]:
        """Update producer state.
        
        Args:
            state: Current agent state
            model_state: Current model state (env + other agents)
            model_config: Model configuration
            key: JAX random key
            
        Returns:
            Updated agent state dictionary
        """
        # Calculate production (Cobb-Douglas production function)
        production = self.productivity * (state['capital'] ** 0.7)
        
        # Calculate revenue (production * price_level from model env state)
        revenue = production * model_state['env']['price_level']
        
        # Calculate costs (fixed capital cost + variable cost)
        costs = 0.1 * state['capital'] + 0.05 * production
        
        # Calculate profit
        profit = revenue - costs
        
        # Update capital (reinvest some profit)
        capital = state['capital'] + self.reinvestment_rate * profit
        
        # Return updated state
        return {
            'capital': capital,
            'production': production,
            'profit': profit
        }


# Define model update and metrics functions according to the new API

def professional_update_state(env_state: Dict[str, Any], agent_states: Dict[str, Dict[str, Any]], 
                              model_params: Dict[str, Any], key: jax.Array) -> Dict[str, Any]:
    """Update the environment state based on agent states and model parameters."""
    
    # Extract agent states
    consumer_states = agent_states.get('consumers')
    producer_states = agent_states.get('producers')
    
    # Calculate total consumption and production from agent states
    total_consumption = jnp.sum(consumer_states['consumption']) if consumer_states else jnp.array(0.0)
    total_production = jnp.sum(producer_states['production']) if producer_states else jnp.array(0.0)
    
    # Update price level based on supply and demand using model params
    price_diff = total_consumption - total_production
    relative_diff = price_diff / (jnp.maximum(total_consumption, total_production) + 1e-8)
    price_adjustment = jnp.clip(relative_diff, -0.5, 0.5) # Limit extreme adjustments
    
    price_level = env_state['price_level'] * (1.0 + model_params['price_adjustment_rate'] * price_adjustment)
    price_level = jnp.maximum(price_level, 0.1) # Ensure price level stays positive
    
    # Calculate market subsidy using model params
    market_subsidy = model_params['subsidy_rate'] * jnp.maximum(0.0, 
        price_level - model_params['target_price']
    )
    
    # Update GDP
    gdp = total_production * price_level
    
    # Update unemployment rate (simplified proxy)
    unemployment = jnp.maximum(0.0, jnp.minimum(1.0, 
        1.0 - total_production / (total_consumption + 1e-8)
    ))
    
    # Return the new environment state dictionary
    # Keep model params in env_state if they can change (they are static here)
    return {
        'price_level': price_level,
        'market_subsidy': market_subsidy,
        'gdp': gdp,
        'unemployment': unemployment,
        'total_consumption': total_consumption,
        'total_production': total_production
    }

def professional_compute_metrics(env_state: Dict[str, Any], agent_states: Dict[str, Dict[str, Any]], 
                                 model_params: Dict[str, Any]) -> Dict[str, Any]:
    """Compute metrics from the current environment and agent states."""
    
    consumer_states = agent_states.get('consumers')
    producer_states = agent_states.get('producers')

    avg_utility = jnp.mean(consumer_states['utility']) if consumer_states else jnp.array(0.0)
    avg_profit = jnp.mean(producer_states['profit']) if producer_states else jnp.array(0.0)
    price_stability = 1.0 / (jnp.abs(env_state['price_level'] - model_params['target_price']) + 1e-8)
    
    if consumer_states and 'consumption' in consumer_states:
        cons = consumer_states['consumption']
        inequality = jnp.max(cons) / (jnp.min(cons) + 1e-8) if cons.size > 0 else jnp.array(1.0)
    else:
        inequality = jnp.array(1.0)

    return {
        'gdp': env_state['gdp'],
        'unemployment': env_state['unemployment'],
        'avg_utility': avg_utility,
        'avg_profit': avg_profit,
        'price_stability': price_stability,
        'inequality': inequality
    }


def create_economy_model(
    num_consumers=50, 
    num_producers=10, 
    income_level=1.0,
    propensity_to_consume=0.8,
    productivity=1.0,
    capital_level=10.0,
    price_adjustment_rate=0.1,
    subsidy_rate=0.2,
    target_price=1.0,
    seed=0,
    # Parameters for analysis tools
    params=None,
    config=None
):
    """Create an economy model with the given parameters.
    
    Args:
        num_consumers: Number of consumer agents
        num_producers: Number of producer agents
        income_level: Base income level for consumers
        propensity_to_consume: Consumer propensity to consume
        productivity: Producer productivity factor
        capital_level: Base capital level for producers
        price_adjustment_rate: Rate of price adjustment to supply/demand
        subsidy_rate: Government subsidy rate
        target_price: Government target price
        seed: Random seed
        params: Dictionary of parameters (used by analysis tools)
        config: ModelConfig object (used by analysis tools)
        
    Returns:
        Initialized Model instance
    """
    # If params is provided, extract parameters from it
    if params is not None:
        propensity_to_consume = params.get('propensity_to_consume', propensity_to_consume)
        productivity = params.get('productivity', productivity)
        subsidy_rate = params.get('subsidy_rate', subsidy_rate)
        
    # If config is provided, use it instead of creating a new one
    if config is None:
        config = ModelConfig(seed=seed)
        
    # Create agent type instances
    consumer_agent_type = Consumer(income_level=income_level, propensity_to_consume=propensity_to_consume)
    producer_agent_type = Producer(productivity=productivity, capital_level=capital_level)

    # Create agent collections
    consumers = AgentCollection(
        agent_type=consumer_agent_type,
        num_agents=num_consumers
    )
    producers = AgentCollection(
        agent_type=producer_agent_type,
        num_agents=num_producers
    )
    
    # Define initial environment state (excluding model params)
    initial_env_state = {
        'price_level': jnp.array(1.0),
        'market_subsidy': jnp.array(0.0),
        'gdp': jnp.array(0.0),
        'unemployment': jnp.array(0.0),
        'total_consumption': jnp.array(0.0),
        'total_production': jnp.array(0.0)
    }

    # Define model parameters (distinct from env state)
    model_params = {
        'price_adjustment_rate': jnp.array(price_adjustment_rate),
        'subsidy_rate': jnp.array(subsidy_rate),
        'target_price': jnp.array(target_price)
    }

    # Create model config
    config = ModelConfig(seed=seed) # Steps are now passed to run()

    # Create model instance
    model = Model(
        params=model_params,
        config=config,
        update_state_fn=professional_update_state,
        metrics_fn=professional_compute_metrics
    )
    
    # Add agent collections
    model.add_agent_collection('consumers', consumers)
    model.add_agent_collection('producers', producers)
    
    # Add initial environment state
    for name, value in initial_env_state.items():
        model.add_env_state(name, value)

    # Initialization is handled by model.run() or explicit model.initialize()
    # No need for add_step_function anymore
    return model


def run_simulation(args):
    """Run a basic model simulation.
    
    Args:
        args: Command-line arguments
    """
    print("\nRunning economic simulation...")
    
    # Create and run model
    model = create_economy_model(
        num_consumers=args.consumers,
        num_producers=args.producers,
        seed=args.seed
    )
    
    # Model.run() now directly returns the metrics dictionary
    results = model.run(args.steps)
    
    # Print final metrics
    if results and 'gdp' in results and results['gdp']:
        print("\nFinal metrics:")
        for name, values in results.items():
            if name != 'step' and values:
                print(f"  {name}: {values[-1]:.4f}")
    
    # Plot results
    if args.plot:
        plot_results(results)
    
    return results


def run_sensitivity_analysis(args):
    """Run sensitivity analysis.
    
    Args:
        args: Command-line arguments
    """
    print("\nRunning sensitivity analysis...")
    
    # Create model factory function for sensitivity analysis
    def model_factory(**kwargs):
        # Create the model instance using the updated function
        model = create_economy_model(
            num_consumers=args.consumers,
            num_producers=args.producers,
            seed=args.seed,
            **kwargs
        )
        # Run the model and get the metrics dictionary directly
        results = model.run(args.steps)
        
        # Return metrics dictionary
        return results
    
    # Initialize sensitivity analysis
    sensitivity = SensitivityAnalysis(
        model_factory=model_factory,
        param_ranges={
            'propensity_to_consume': (0.6, 0.9),
            'productivity': (0.5, 1.5),
            'subsidy_rate': (0.1, 0.4)
        },
        metrics_of_interest=['gdp', 'unemployment', 'inequality', 'avg_utility'],
        num_samples=args.samples,
        seed=args.seed
    )
    
    # Run analysis
    results = sensitivity.run(verbose=True)
    
    # Calculate and print indices
    indices = sensitivity.sobol_indices()
    
    print("\nSensitivity analysis results (sensitivity indices):")
    for metric, params in indices.items():
        print(f"  {metric}:")
        for param, index in params.items():
            print(f"    {param}: {index:.4f}")
    
    # Plot results
    if args.plot:
        fig, ax = sensitivity.plot_indices()
        plt.savefig('sensitivity_indices.png')
        plt.close()
        print("\nSensitivity plot saved to 'sensitivity_indices.png'")
    
    return indices


def run_calibration(args):
    """Run model calibration.
    
    Args:
        args: Command-line arguments
    """
    print("\nRunning model calibration (this may take several minutes)...")
    
    # Create a smaller model factory function for calibration (for faster gradient computation)
    def calibration_model_factory(**kwargs):
        """Create a smaller model for calibration purposes."""
        # Create the model instance with fewer agents for faster gradient computation
        model = create_economy_model(
            num_consumers=min(args.consumers, 20),  # Limit to at most 20 consumers for speed
            num_producers=min(args.producers, 3),   # Limit to at most 3 producers for speed
            seed=args.seed,
            **kwargs
        )
        # Return the model object itself, not the result of running it
        return model
    
    # Define target metrics
    target_metrics = {
        'gdp': 10.0,
        'unemployment': 0.05,
        'inequality': 2.0
    }
    
    # Initialize calibrator
    calibrator = ModelCalibrator(
        model_factory=calibration_model_factory,  # Use the smaller model factory
        initial_params={
            'propensity_to_consume': 0.7,
            'productivity': 1.0,
            'subsidy_rate': 0.1
        },
        target_metrics=target_metrics,
        metrics_weights={
            'gdp': 0.1,
            'unemployment': 1.0,
            'inequality': 0.5
        },
        learning_rate=0.01,
        max_iterations=args.iterations,
        method=args.method
    )
    
    # Run calibration
    optimal_params = calibrator.calibrate(verbose=True)
    
    # Plot results
    if args.plot:
        fig, axes = calibrator.plot_calibration()
        plt.savefig('calibration_results.png')
        plt.close()
        print("\nCalibration plot saved to 'calibration_results.png'")
    
    # Create and run the full model with calibrated parameters 
    model = create_economy_model(
        num_consumers=args.consumers,
        num_producers=args.producers,
        propensity_to_consume=optimal_params['propensity_to_consume'],
        productivity=optimal_params['productivity'],
        subsidy_rate=optimal_params['subsidy_rate'],
        seed=args.seed
    )
    results_calibrated = model.run(args.steps)
    
    # Print final metrics
    if results_calibrated and 'gdp' in results_calibrated and results_calibrated['gdp']:
        print("\nFinal metrics with calibrated parameters:")
        for name, values in results_calibrated.items():
            if name != 'step' and values:
                if name in target_metrics:
                    print(f"  {name}: {values[-1]:.4f} (target: {target_metrics[name]:.4f})")
                else:
                    print(f"  {name}: {values[-1]:.4f}")
    
    return optimal_params


def plot_results(results):
    """Plot simulation results.
    
    Args:
        results: Dictionary of simulation results
    """
    if not results or 'step' not in results or not results['step']:
        print("No results to plot")
        return
        
    plt.figure(figsize=(12, 8))
    
    # Plot key metrics
    metrics = ['gdp', 'unemployment', 'avg_utility', 'inequality']
    for i, metric in enumerate(metrics):
        if metric in results and results[metric]:
            plt.subplot(2, 2, i+1)
            plt.plot(results['step'], results[metric])
            plt.title(f'{metric.capitalize()} over time')
            plt.xlabel('Step')
            plt.ylabel(metric.capitalize())
            plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('simulation_results.png')
    plt.close()
    print("\nSimulation plot saved to 'simulation_results.png'")


def main():
    """Main function to parse arguments and run examples."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run professional agent-based model examples.")
    parser.add_argument("--simulation", action="store_true", help="Run basic simulation")
    parser.add_argument("--sensitivity", action="store_true", help="Run sensitivity analysis")
    parser.add_argument("--calibration", action="store_true", help="Run model calibration (slow)")
    parser.add_argument("--steps", type=int, default=50, help="Number of simulation steps")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples for sensitivity analysis")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations for calibration")
    parser.add_argument("--method", choices=["gradient", "rl"], default="gradient", help="Calibration method")
    parser.add_argument("--consumers", type=int, default=100, help="Number of consumer agents")
    parser.add_argument("--producers", type=int, default=10, help="Number of producer agents")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--fast", action="store_true", help="Run with minimal agents and steps (faster)")
    parser.add_argument("--all", action="store_true", help="Run all examples")
    
    args = parser.parse_args()
    
    # Modify parameters if fast mode is enabled
    if args.fast:
        args.consumers = min(args.consumers, 20)
        args.producers = min(args.producers, 5)
        args.steps = min(args.steps, 20)
        args.samples = min(args.samples, 3)
        args.iterations = min(args.iterations, 3)
    
    # If no specific example is selected, run simulation by default
    if not any([args.simulation, args.sensitivity, args.calibration, args.all]):
        args.simulation = True
    
    # Run all examples if --all is specified
    if args.all:
        args.simulation = True
        args.sensitivity = True
        # Calibration is still optional due to slowness
        # args.calibration = True
    
    # Run examples based on arguments
    if args.simulation:
        run_simulation(args)
    
    if args.sensitivity:
        run_sensitivity_analysis(args)
    
    if args.calibration:
        # For calibration, use even smaller numbers to make it faster
        calibration_args = argparse.Namespace(**vars(args))
        if not args.fast:
            # If not already in fast mode, use reduced parameters for calibration
            calibration_args.consumers = min(args.consumers, 50)
            calibration_args.producers = min(args.producers, 5)
            calibration_args.steps = min(args.steps, 30)
        run_calibration(calibration_args)
        
    # No examples were actually run (would only happen with --calibration but it was skipped)
    if not any([args.simulation, args.sensitivity, args.calibration]):
        print("\nNo examples were run. Try --simulation, --sensitivity, or --calibration.")
        print("For faster execution: --fast")
        print("For all examples: --all")
        parser.print_help()


if __name__ == "__main__":
    main() 