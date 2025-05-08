"""
Simplified example of the JAX-based ABM for faster execution.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from typing import Dict, Any

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
    num_consumers=20,  # Reduced number of agents
    num_producers=5,   # Reduced number of agents
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
    
    # If config is provided, use it; otherwise create a new one
    if config is None:
        config = ModelConfig(
            steps=20,  # Reduced number of steps
            collect_interval=1,
            seed=seed
        )
    
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


def run_basic_simulation():
    """Run a basic model simulation."""
    print("\n1. Running basic model simulation...")
    model = create_economy_model(seed=42)
    results = model.run(steps=20)
    
    # Print some results
    print("\nFinal metrics:")
    for metric in ['gdp', 'unemployment', 'avg_utility', 'inequality']:
        print(f"  {metric}: {results[metric][-1]:.4f}")
    
    return results


def run_calibration():
    """Run a calibration example."""
    print("\n2. Running model calibration...")
    
    # Define target metrics with more realistic values
    target_metrics = {
        'gdp': 10.0,             # More realistic target (typical values are 3-5)
        'unemployment': 0.05,
        'inequality': 2.0        # More realistic target (typical values are 1.4-1.5)
    }
    
    # Create a simplified model factory for calibration (even fewer agents for faster gradient computation)
    def calibration_model_factory(**kwargs):
        # Use a very reduced model for calibration to speed things up
        model = create_economy_model(
            num_consumers=10,  # Very few agents for faster gradient computation
            num_producers=2,   # Very few agents
            seed=42,
            **kwargs
        )
        # Return the model object itself, not the result of running it
        return model
    
    # Initialize calibrator
    calibrator_gd = ModelCalibrator(
        model_factory=calibration_model_factory,  # Use the simplified factory
        initial_params={
            'propensity_to_consume': 0.7,
            'productivity': 1.0,
            'subsidy_rate': 0.1
        },
        target_metrics=target_metrics,
        metrics_weights={
            'gdp': 0.1,           # Reduced weight to prevent extreme gradient values
            'unemployment': 1.0,
            'inequality': 0.5
        },
        learning_rate=0.01,       # Reduced learning rate for more stable optimization
        max_iterations=3,         # Fewer iterations for testing
        method='gradient'
    )
    
    # Run calibration
    optimal_params = calibrator_gd.calibrate()
    return optimal_params


def run_sensitivity():
    """Run a sensitivity analysis example."""
    print("\n3. Running sensitivity analysis...")
    
    # Create a wrapper function for the sensitivity analysis
    def sensitivity_model_factory(params=None, config=None):
        """Wrapper for create_economy_model with appropriate defaults."""
        return create_economy_model(
            num_consumers=10,  # Use fewer agents for faster analysis
            num_producers=2,   # Use fewer agents
            params=params,
            config=config
        )
    
    # Initialize sensitivity analysis
    sensitivity = SensitivityAnalysis(
        model_factory=sensitivity_model_factory,
        param_ranges={
            'propensity_to_consume': (0.6, 0.9),
            'productivity': (0.5, 1.5),
        },
        metrics_of_interest=['gdp', 'unemployment', 'inequality'],
        num_samples=5,  # Just a few samples for demonstration
        seed=42
    )
    
    # Run sensitivity analysis
    results = sensitivity.run()
    
    # Calculate and print Sobol indices
    sobol_indices = sensitivity.sobol_indices()
    
    print("\nSensitivity analysis results (Sobol indices):")
    for metric, indices in sobol_indices.items():
        print(f"  {metric}:")
        for param, index in indices.items():
            print(f"    {param}: {index:.4f}")
    
    return results


def run_multiple_agent_types():
    """Demonstrate running a model with different agent types."""
    print("\n4. Running model with multiple agent types...")
    
    # Create a model with different configurations of agents
    model = create_economy_model(
        num_consumers=20,      # Regular consumers
        num_producers=5,       # Regular producers
        propensity_to_consume=0.8,
        productivity=1.0,
        seed=42
    )
    
    # Run the model
    print("Running the model...")
    results = model.run(steps=20)
    
    # Print summary statistics about different agent types
    print("\nResults from multiple agent types:")
    
    # Access the latest metrics
    print("\nFinal metrics:")
    for metric in ['gdp', 'unemployment', 'avg_utility', 'inequality', 'avg_profit']:
        if metric in results:
            print(f"  {metric}: {results[metric][-1]:.4f}")
    
    # Access the agent collections
    consumers = model.agent_collections['consumers']
    producers = model.agent_collections['producers']
    
    # Print stats about agent counts
    print("\nAgent counts:")
    print(f"  Consumers: {consumers.num_agents}")
    print(f"  Producers: {producers.num_agents}")
    
    # Print final model state
    print("\nFinal model state:")
    for key, value in model.state.items():
        if hasattr(value, 'item'):
            print(f"  {key}: {value.item():.4f}")
        else:
            print(f"  {key}: {value}")
    
    return results


def main():
    """Run the simplified example."""
    print("Running simplified JAX-based agent model example...")
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "calibrate":
            # Run only calibration
            print("Running calibration directly...")
            run_calibration()
            return
        elif sys.argv[1] == "sensitivity":
            # Run only sensitivity analysis
            print("Running sensitivity analysis directly...")
            run_sensitivity()
            return
        elif sys.argv[1] == "multi":
            # Run multiple agent types example
            print("Running multiple agent types example...")
            run_multiple_agent_types()
            return
        
    # Run basic simulation
    run_basic_simulation()
    
    # Ask user what to run next
    print("\nWhat would you like to run next?")
    print("1. Calibration")
    print("2. Sensitivity Analysis")
    print("3. Multiple Agent Types")
    print("4. Exit")
    choice = input("Enter your choice (1-4): ")
    
    if choice == "1":
        run_calibration()
    elif choice == "2":
        run_sensitivity()
    elif choice == "3":
        run_multiple_agent_types()
    
    print("\nExample completed.")


if __name__ == "__main__":
    main() 