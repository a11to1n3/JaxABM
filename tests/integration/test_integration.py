"""
Integration tests for the JaxABM framework.

These tests verify the integration of different components of the JaxABM
framework and test complete workflows.
"""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any

from jaxabm.agent import AgentType, AgentCollection
from jaxabm.core import ModelConfig
from jaxabm.model import Model
from jaxabm.analysis import SensitivityAnalysis, ModelCalibrator


# Define a simple economy model for integration testing
class Consumer(AgentType):
    """Consumer agent that earns income and purchases goods."""
    
    def __init__(self, base_income=1.0, propensity_to_consume=0.8):
        """Initialize Consumer agent type.
        
        Args:
            base_income: Base income for all consumers
            propensity_to_consume: Proportion of income spent on consumption
        """
        self.base_income = base_income
        self.propensity_to_consume = propensity_to_consume

    def init_state(self, model_config: ModelConfig, key: jax.Array) -> Dict[str, Any]:
        """Initialize consumer state."""
        income = self.base_income * (0.8 + 0.4 * jax.random.uniform(key))
        return {
            'savings': jnp.array(0.0),
            'consumption': jnp.array(0.0),
            'utility': jnp.array(0.0),
            'income': jnp.array(income)
        }
    
    def update(self, state: Dict[str, Any], model_state: Dict[str, Any], 
               model_config: ModelConfig, key: jax.Array) -> Dict[str, Any]:
        """Update consumer state."""
        # Calculate consumption based on propensity to consume
        price_level = model_state['env'].get('price_level', 1.0)
        
        # Adjust consumption based on price level
        consumption = self.propensity_to_consume * state['income'] / price_level
        
        # Calculate savings
        savings = state['savings'] + (state['income'] - consumption * price_level)
        
        # Calculate utility (logarithmic utility function)
        utility = jnp.log(consumption + 1.0)
        
        # Update state
        new_state = {
            'savings': savings,
            'consumption': consumption,
            'utility': utility,
            'income': state['income']
        }
        
        # Return new state only
        return new_state


class Producer(AgentType):
    """Producer agent that produces goods and services."""
    
    def __init__(self, initial_capital=10.0, productivity=1.0, reinvestment_rate=0.3):
        """Initialize Producer agent type.
        
        Args:
            initial_capital: Initial capital for producers
            productivity: Production efficiency parameter
            reinvestment_rate: Proportion of profits reinvested
        """
        self.initial_capital = initial_capital
        self.productivity = productivity
        self.reinvestment_rate = reinvestment_rate

    def init_state(self, model_config: ModelConfig, key: jax.Array) -> Dict[str, Any]:
        """Initialize producer state."""
        capital = self.initial_capital * (0.8 + 0.4 * jax.random.uniform(key))
        return {
            'capital': jnp.array(capital),
            'production': jnp.array(0.0),
            'profit': jnp.array(0.0)
        }
    
    def update(self, state: Dict[str, Any], model_state: Dict[str, Any], 
               model_config: ModelConfig, key: jax.Array) -> Dict[str, Any]:
        """Update producer state."""
        # Calculate production
        production = self.productivity * (state['capital'] ** 0.7)
        
        # Calculate revenue
        price_level = model_state['env'].get('price_level', 1.0)
        revenue = production * price_level
        
        # Calculate costs
        costs = 0.1 * state['capital'] + 0.05 * production
        
        # Calculate profit
        profit = revenue - costs
        
        # Update capital (reinvest profits)
        capital = state['capital'] + self.reinvestment_rate * profit
        
        # Update state
        new_state = {
            'capital': capital,
            'production': production,
            'profit': profit
        }
        
        # Return new state only
        return new_state


# Define model update and metrics functions
def update_model_state(env_state: Dict[str, Any], agent_states: Dict[str, Dict[str, Any]], 
                       params: Dict[str, Any], key: jax.Array) -> Dict[str, Any]:
    """Update the model state."""
    # Extract agent states
    consumer_states = agent_states.get('consumers')
    producer_states = agent_states.get('producers')
    
    # Get total consumption and production
    total_consumption = jnp.sum(consumer_states['consumption']) if consumer_states else jnp.array(0.0)
    total_production = jnp.sum(producer_states['production']) if producer_states else jnp.array(0.0)
    
    # Adjust price level based on supply and demand using model params
    price_adjustment_rate = params.get('price_adjustment_rate', 0.1)
    supply_demand_ratio = (total_production + 1e-8) / (total_consumption + 1e-8)
    price_change = price_adjustment_rate * (1.0 - supply_demand_ratio)
    
    # Update price level
    price_level = env_state.get('price_level', 1.0) * (1.0 + price_change)
    price_level = jnp.clip(price_level, 0.5, 2.0)  # Keep within reasonable bounds
    
    # Calculate GDP
    gdp = total_production * price_level
    
    # Calculate unemployment rate (proxy)
    unemployment = jnp.maximum(0.0, jnp.minimum(0.5, 1.0 - supply_demand_ratio))
    
    # Update state
    new_env_state = {
        'price_level': price_level,
        'gdp': gdp,
        'unemployment': unemployment,
        'total_consumption': total_consumption,
        'total_production': total_production
    }
    
    return new_env_state


def compute_metrics(env_state: Dict[str, Any], agent_states: Dict[str, Dict[str, Any]], 
                    params: Dict[str, Any]) -> Dict[str, Any]:
    """Compute model metrics."""
    # Economic metrics from env state
    metrics = {
        'gdp': env_state.get('gdp', 0.0),
        'price_level': env_state.get('price_level', 1.0),
        'unemployment': env_state.get('unemployment', 0.0)
    }
    
    # Agent-based metrics from agent states
    consumer_states = agent_states.get('consumers')
    producer_states = agent_states.get('producers')
    
    if consumer_states and 'utility' in consumer_states:
        metrics['avg_utility'] = jnp.mean(consumer_states['utility'])
    
    if producer_states and 'profit' in producer_states:
        metrics['avg_profit'] = jnp.mean(producer_states['profit'])
    
    return metrics


# Factory function for creating the economy model
def create_economy_model(
    # Agent counts
    num_consumers=20,
    num_producers=5,
    # Agent parameters (can be overridden)
    base_income=1.0,
    propensity_to_consume=0.8,
    initial_capital=10.0,
    productivity=1.0,
    reinvestment_rate=0.3,
    # Model parameters (can be overridden)
    price_adjustment_rate=0.1,
    target_price=1.0, # Note: target_price is a model param but not used in the new update_model_state
    # Simulation config
    seed=0,
    # Parameters for analysis tools
    params=None,
    config=None
):
    """Create an economy model for testing using the new API.
    
    This function can be called directly with specific parameters or via
    analysis tools that provide a params dictionary.
    
    Args:
        num_consumers: Number of consumer agents
        num_producers: Number of producer agents
        base_income: Base income for consumers
        propensity_to_consume: Proportion of income spent on consumption
        initial_capital: Initial capital for producers
        productivity: Production efficiency parameter
        reinvestment_rate: Proportion of profits reinvested
        price_adjustment_rate: Rate at which prices adjust
        target_price: Target price level
        seed: Random seed for reproducibility
        params: Dictionary of parameters (used by analysis tools)
        config: ModelConfig object (used by analysis tools)
    
    Returns:
        Initialized Model object
    """
    # If params is provided (by analysis tools), use those parameters
    if params is not None:
        propensity_to_consume = params.get('propensity_to_consume', propensity_to_consume)
        productivity = params.get('productivity', productivity)
        price_adjustment_rate = params.get('price_adjustment_rate', price_adjustment_rate)
    
    # If config is provided, use it instead of creating a new one
    if config is None:
        config = ModelConfig(seed=seed)
    
    # Create agent type instances
    consumer_agent_type = Consumer(base_income=base_income, 
                               propensity_to_consume=propensity_to_consume)
    producer_agent_type = Producer(initial_capital=initial_capital, 
                               productivity=productivity, 
                               reinvestment_rate=reinvestment_rate)

    # Create agent collections
    consumers = AgentCollection(
        agent_type=consumer_agent_type,
        num_agents=num_consumers
    )
    producers = AgentCollection(
        agent_type=producer_agent_type,
        num_agents=num_producers
    )
    
    # Initial environment state
    initial_env_state = {
        'price_level': 1.0,
        'gdp': 0.0,
        'unemployment': 0.0,
        'total_consumption': 0.0,
        'total_production': 0.0
    }
    
    # Model parameters
    model_params = {
        'price_adjustment_rate': price_adjustment_rate,
        'target_price': target_price # Passed but not used in new update_fn
    }
    
    # Create model instance
    model = Model(
        params=model_params,
        config=config,
        update_state_fn=update_model_state,
        metrics_fn=compute_metrics,
    )

    # Add collections and initial state
    model.add_agent_collection('consumers', consumers)
    model.add_agent_collection('producers', producers)
    for name, value in initial_env_state.items():
        model.add_env_state(name, value)

    return model


class TestCompleteWorkflow(unittest.TestCase):
    """Integration tests for complete workflows with the JaxABM framework."""
    
    def test_basic_simulation(self):
        """Test running a basic simulation."""
        # Create model
        model = create_economy_model(seed=42)
        
        # Run model
        results = model.run()
        
        # Check that we have the expected metrics
        self.assertIn('gdp', results)
        self.assertIn('price_level', results)
        self.assertIn('unemployment', results)
        self.assertIn('avg_utility', results)
        self.assertIn('avg_profit', results)
        
        # Check that metrics are stored for every time step
        self.assertEqual(len(results['gdp']), model.config.steps)
        
        # Values should have changed over time
        self.assertNotEqual(results['gdp'][0], results['gdp'][-1])
        self.assertNotEqual(results['price_level'][0], results['price_level'][-1])
    
    def test_parameter_variation(self):
        """Test running simulations with different parameters."""
        # Baseline model
        baseline_model = create_economy_model(
            propensity_to_consume=0.8,
            productivity=1.0,
            seed=42
        )
        baseline_results = baseline_model.run()
        
        # High propensity to consume
        high_consumption_model = create_economy_model(
            propensity_to_consume=0.9,
            productivity=1.0,
            seed=42
        )
        high_consumption_results = high_consumption_model.run()
        
        # High productivity
        high_productivity_model = create_economy_model(
            propensity_to_consume=0.8,
            productivity=1.2,
            seed=42
        )
        high_productivity_results = high_productivity_model.run()
        
        # Check that higher propensity to consume leads to higher GDP
        self.assertGreater(
            high_consumption_results['gdp'][-1],
            baseline_results['gdp'][-1]
        )
        
        # Check that higher productivity leads to higher GDP
        self.assertGreater(
            high_productivity_results['gdp'][-1],
            baseline_results['gdp'][-1]
        )
    
    def test_calibration(self):
        """Test model calibration."""
        # Define target metrics
        target_metrics = {
            'gdp': 100.0,
            'price_level': 1.2
        }
        
        # Initial parameters
        initial_params = {
            'propensity_to_consume': 0.7,
            'productivity': 0.9
        }
        
        # Create calibrator
        calibrator = ModelCalibrator(
            model_factory=create_economy_model,
            initial_params=initial_params,
            target_metrics=target_metrics,
            max_iterations=2  # Very small for testing
        )
        
        # Run calibration
        optimized_params = calibrator.calibrate(verbose=False)
        
        # Check that we got parameters back
        self.assertIn('propensity_to_consume', optimized_params)
        self.assertIn('productivity', optimized_params)
        
        # Run model with optimized parameters
        model = create_economy_model(
            propensity_to_consume=optimized_params['propensity_to_consume'],
            productivity=optimized_params['productivity'],
            seed=42
        )
        results = model.run()
        
        # Check that we're at least moving toward the targets
        initial_model = create_economy_model(**initial_params, seed=42)
        initial_results = initial_model.run()
        
        # Calculate distances to targets
        initial_gdp_dist = abs(initial_results['gdp'][-1] - target_metrics['gdp'])
        optimized_gdp_dist = abs(results['gdp'][-1] - target_metrics['gdp'])
        
        initial_price_dist = abs(initial_results['price_level'][-1] - target_metrics['price_level'])
        optimized_price_dist = abs(results['price_level'][-1] - target_metrics['price_level'])
        
        # At least one metric should be closer to target
        self.assertTrue(
            optimized_gdp_dist < initial_gdp_dist or
            optimized_price_dist < initial_price_dist
        )
    
    def test_sensitivity_analysis(self):
        """Test sensitivity analysis."""
        # Define parameter ranges
        param_ranges = {
            'propensity_to_consume': (0.7, 0.9),
            'productivity': (0.8, 1.2)
        }
        
        # Define metrics of interest
        metrics = ['gdp', 'price_level', 'avg_utility']
        
        # Create sensitivity analysis
        sa = SensitivityAnalysis(
            model_factory=create_economy_model,
            param_ranges=param_ranges,
            metrics_of_interest=metrics,
            num_samples=3  # Very small for testing
        )
        
        # Run analysis
        results = sa.run(verbose=False)
        
        # Check that we have results for each metric
        for metric in metrics:
            self.assertIn(metric, results)
            self.assertEqual(len(results[metric]), 3)
        
        # Calculate Sobol indices
        indices = sa.sobol_indices()
        
        # Check that we have indices for each metric
        for metric in metrics:
            self.assertIn(metric, indices)
            for param in param_ranges:
                self.assertIn(param, indices[metric])


class TestEndToEndWorkflow(unittest.TestCase):
    """End-to-end tests that combine multiple components."""
    
    def test_end_to_end(self):
        """Test an end-to-end workflow combining all components."""
        # 1. Create and run base model
        model = create_economy_model(seed=42)
        results = model.run()
        
        # 2. Run sensitivity analysis
        param_ranges = {
            'propensity_to_consume': (0.7, 0.9),
            'productivity': (0.8, 1.2),
            'price_adjustment_rate': (0.05, 0.2)
        }
        metrics = ['gdp', 'price_level', 'unemployment']
        
        sa = SensitivityAnalysis(
            model_factory=create_economy_model,
            param_ranges=param_ranges,
            metrics_of_interest=metrics,
            num_samples=3  # Very small for testing
        )
        sa_results = sa.run(verbose=False)
        
        # 3. Identify most sensitive parameter
        indices = sa.sobol_indices()
        gdp_indices = indices['gdp']
        most_sensitive_param = max(gdp_indices, key=gdp_indices.get)
        
        # 4. Calibrate that parameter to a target
        target_metrics = {
            'gdp': 120.0  # Target for GDP
        }
        
        # Use only the most sensitive parameter
        initial_params = {
            most_sensitive_param: 0.8  # Start with a middle value
        }
        
        calibrator = ModelCalibrator(
            model_factory=create_economy_model,
            initial_params=initial_params,
            target_metrics=target_metrics,
            max_iterations=2  # Very small for testing
        )
        
        optimized_params = calibrator.calibrate(verbose=False)
        
        # 5. Run optimized model
        model_params = {}
        model_params[most_sensitive_param] = optimized_params[most_sensitive_param]
        optimized_model = create_economy_model(**model_params, seed=42)
        optimized_results = optimized_model.run()
        
        # Compare outputs - should be different from original run
        self.assertNotEqual(results['gdp'][-1], optimized_results['gdp'][-1])


if __name__ == '__main__':
    unittest.main() 