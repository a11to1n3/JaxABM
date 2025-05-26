"""
Unit tests for the jaxabm.analysis module.
"""
import unittest
import jax
import jax.numpy as jnp
from typing import Dict, Any
import importlib.util

# Check if matplotlib is available
matplotlib_available = importlib.util.find_spec("matplotlib") is not None

from jaxabm.analysis import SensitivityAnalysis, ModelCalibrator
from jaxabm.model import Model
from jaxabm.core import ModelConfig
from jaxabm.agent import AgentCollection, AgentType

from tests.unit.test_agent import TestAgent


# Use the DummyAgent from test_model or define locally
class DummyAgent(AgentType):
    def __init__(self, growth_rate=0.1, initial_value=0.0):
        """Initialize DummyAgent with configurable parameters.
        
        Args:
            growth_rate: Rate at which agent value grows each step
            initial_value: Starting value for agent state
        """
        self.growth_rate = growth_rate
        self.initial_value = initial_value
    
    def init_state(self, model_config: ModelConfig, key: jax.Array) -> Dict[str, Any]:
        return {'value': jnp.array(self.initial_value)}
    
    def update(self, state: Dict[str, Any], model_state: Dict[str, Any], 
               model_config: ModelConfig, key: jax.Array) -> Dict[str, Any]:
        # Grow value based on growth rate
        return {'value': state['value'] * (1.0 + self.growth_rate)}


# Define a simple factory function for creating models
def create_test_model(
    # Parameters that might be varied by analysis tools
    growth_rate=0.1,
    adjustment_rate=0.1,
    # Other fixed parameters for the model setup
    initial_value=0.0,
    num_agents=10,
    seed=0,
    # Special params dict used by analysis tools
    params=None,
    config=None
):
    """Create a simple test model for unit testing using the new API.
    
    This function can be called directly with specific parameters or via
    analysis tools that provide a params dictionary.
    
    Args:
        growth_rate: Growth rate for agent values
        adjustment_rate: Rate of price level adjustment
        initial_value: Initial value for agents
        num_agents: Number of agents to create
        seed: Random seed
        params: Dictionary of parameters (used by analysis tools)
        config: ModelConfig object (used by analysis tools)
    
    Returns:
        Initialized Model object
    """
    # If params is provided (by analysis tools), use it to override defaults
    if params is not None:
        # Extract parameters from the params dictionary
        growth_rate = params.get('growth_rate', growth_rate)
        adjustment_rate = params.get('adjustment_rate', adjustment_rate)
    
    # If config is provided, use it instead of creating a new one
    if config is None:
        config = ModelConfig(seed=seed)
    
    # Create agent type instance with specific parameters for this run
    test_agent_type = DummyAgent(growth_rate=growth_rate, initial_value=initial_value)

    # Create agent collection
    consumers = AgentCollection(
        agent_type=test_agent_type,
        num_agents=num_agents
    )
    
    # Define initial environment state
    initial_env_state = {
        'price_level': 1.0,
        'interest_rate': 0.05,
        # 'value': initial_value # Initial value is now part of agent state
    }
    
    # Define model parameters (can include agent params if needed by update/metrics fn)
    model_params = {
        'adjustment_rate': adjustment_rate,
        'target_price': 1.2
    }
    
    # Simple update function (new signature)
    def update_fn(env_state: Dict[str, Any], agent_states: Dict[str, Dict[str, Any]], 
                  params: Dict[str, Any], key: jax.Array) -> Dict[str, Any]:
        new_env_state = dict(env_state) # Copy
        # Update price level based on adjustment rate
        new_env_state['price_level'] += params['adjustment_rate'] * (params['target_price'] - env_state['price_level'])
        return new_env_state
    
    # Simple metrics function (new signature)
    def metrics_fn(env_state: Dict[str, Any], agent_states: Dict[str, Dict[str, Any]], 
                   params: Dict[str, Any]) -> Dict[str, Any]:
        metrics = {}
        consumer_states = agent_states.get('consumers')
        
        # Average value from agent states
        if consumer_states and 'value' in consumer_states:
            metrics['avg_value'] = jnp.mean(consumer_states['value'])
        
        # Price level from env state
        metrics['price_level'] = env_state['price_level']
        
        # Price gap using env state and params
        metrics['price_gap'] = jnp.abs(env_state['price_level'] - params['target_price'])
        
        return metrics
    
    # Create the model instance
    model = Model(
        params=model_params,
        config=config,
        update_state_fn=update_fn,
        metrics_fn=metrics_fn,
    )

    # Add collections and initial state
    model.add_agent_collection('consumers', consumers)
    for name, value in initial_env_state.items():
        model.add_env_state(name, value)

    # Model is ready, initialization happens within run() or via initialize()
    return model


class TestSensitivityAnalysis(unittest.TestCase):
    """Tests for the SensitivityAnalysis class."""
    
    def setUp(self):
        """Set up test environment."""
        # Define parameter ranges
        self.param_ranges = {
            'growth_rate': (0.05, 0.2),
            'adjustment_rate': (0.05, 0.3)
        }
        
        # Define metrics of interest
        self.metrics = ['avg_value', 'price_level', 'price_gap']
        
        # Small number of samples for testing
        self.num_samples = 5
        
        # Create sensitivity analysis
        self.sa = SensitivityAnalysis(
            model_factory=create_test_model,
            param_ranges=self.param_ranges,
            metrics_of_interest=self.metrics,
            num_samples=self.num_samples,
            seed=42
        )
    
    def test_init(self):
        """Test initialization of SensitivityAnalysis."""
        self.assertEqual(self.sa.model_factory, create_test_model)
        self.assertEqual(self.sa.param_ranges, self.param_ranges)
        self.assertEqual(self.sa.metrics_of_interest, self.metrics)
        self.assertEqual(self.sa.num_samples, self.num_samples)
        
        # Check that samples were generated
        self.assertIsNotNone(self.sa.samples)
        self.assertEqual(self.sa.samples.shape, (self.num_samples, len(self.param_ranges)))
        
        # Check that samples are in the correct range
        for i, (param, (min_val, max_val)) in enumerate(self.param_ranges.items()):
            param_samples = self.sa.samples[:, i]
            self.assertTrue(jnp.all(param_samples >= min_val))
            self.assertTrue(jnp.all(param_samples <= max_val))
    
    def test_run(self):
        """Test running sensitivity analysis."""
        # Run with small number of samples
        results = self.sa.run(verbose=False)
        
        # Check that results have the expected structure
        self.assertIsInstance(results, dict)
        for metric in self.metrics:
            self.assertIn(metric, results)
            self.assertEqual(results[metric].shape, (self.num_samples,))
    
    def test_sobol_indices(self):
        """Test calculating sensitivity indices."""
        # Run first
        self.sa.run(verbose=False)
        
        # Calculate indices
        indices = self.sa.sobol_indices()
        
        # Check that indices have the expected structure
        self.assertIsInstance(indices, dict)
        for metric in self.metrics:
            self.assertIn(metric, indices)
            self.assertIsInstance(indices[metric], dict)
            for param in self.param_ranges:
                self.assertIn(param, indices[metric])
                self.assertIsInstance(indices[metric][param], float)
        
        # Test without running first - should raise
        sa = SensitivityAnalysis(
            model_factory=create_test_model,
            param_ranges=self.param_ranges,
            metrics_of_interest=self.metrics,
            num_samples=self.num_samples
        )
        with self.assertRaises(ValueError):
            sa.sobol_indices()
    
    @unittest.skipIf(not matplotlib_available, "Matplotlib not available")
    def test_plot_indices(self):
        """Test plotting sensitivity indices."""
        # Run first
        self.sa.run(verbose=False)
        
        # Try plotting
        fig, ax = self.sa.plot_indices()
        
        # Check that we got matplotlib objects
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)


class TestModelCalibrator(unittest.TestCase):
    """Tests for the ModelCalibrator class."""
    
    def setUp(self):
        """Set up test environment."""
        # Define initial parameters
        self.initial_params = {
            'growth_rate': 0.1,
            'adjustment_rate': 0.1,
        }
        
        # Define target metrics
        self.target_metrics = {
            'avg_value': 2.0,
            'price_level': 1.2
        }
        
        # Define weights
        self.weights = {
            'avg_value': 1.0,
            'price_level': 2.0
        }
        
        # Create calibrator with gradient method (Adam)
        self.calibrator_gradient = ModelCalibrator(
            model_factory=create_test_model,
            initial_params=self.initial_params,
            target_metrics=self.target_metrics,
            metrics_weights=self.weights,
            learning_rate=0.05,
            max_iterations=3,  # Small for testing
            method="adam"
        )
        
        # Create calibrator with RL method (Q-learning)
        self.calibrator_rl = ModelCalibrator(
            model_factory=create_test_model,
            initial_params=self.initial_params,
            target_metrics=self.target_metrics,
            metrics_weights=self.weights,
            learning_rate=0.05,
            max_iterations=3,  # Small for testing
            method="q_learning"
        )
    
    def test_init(self):
        """Test initialization of ModelCalibrator."""
        # Gradient method
        self.assertEqual(self.calibrator_gradient.model_factory, create_test_model)
        self.assertEqual(self.calibrator_gradient.params, self.initial_params)
        self.assertEqual(self.calibrator_gradient.target_metrics, self.target_metrics)
        self.assertEqual(self.calibrator_gradient.metrics_weights, self.weights)
        self.assertEqual(self.calibrator_gradient.learning_rate, 0.05)
        self.assertEqual(self.calibrator_gradient.max_iterations, 3)
        self.assertEqual(self.calibrator_gradient.method, "adam")
        
        # RL method
        self.assertEqual(self.calibrator_rl.method, "q_learning")
        
        # Invalid method should raise
        with self.assertRaises(ValueError):
            ModelCalibrator(
                model_factory=create_test_model,
                initial_params=self.initial_params,
                target_metrics=self.target_metrics,
                method="invalid_method"
            )
    
    def test_gradient_calibration(self):
        """Test gradient-based calibration."""
        # Run calibration with limited iterations
        optimized_params = self.calibrator_gradient.calibrate(verbose=False)
        
        # Check that we got parameters back
        self.assertIsInstance(optimized_params, dict)
        for param, value in self.initial_params.items():
            self.assertIn(param, optimized_params)
        
        # Check that history was recorded
        self.assertEqual(len(self.calibrator_gradient.loss_history), 3)
        self.assertEqual(len(self.calibrator_gradient.param_history), 3)  # 3 iterations
    
    def test_rl_calibration(self):
        """Test RL-based calibration."""
        # Run calibration with limited iterations
        optimized_params = self.calibrator_rl.calibrate(verbose=False)
        
        # Check that we got parameters back
        self.assertIsInstance(optimized_params, dict)
        for param, value in self.initial_params.items():
            self.assertIn(param, optimized_params)
        
        # Check that history was recorded
        self.assertEqual(len(self.calibrator_rl.loss_history), 3)
        self.assertEqual(len(self.calibrator_rl.param_history), 3)  # 3 iterations
    
    def test_get_calibration_history(self):
        """Test getting calibration history."""
        # Run calibration first
        self.calibrator_gradient.calibrate(verbose=False)
        
        # Get history
        history = self.calibrator_gradient.get_calibration_history()
        
        # Check structure
        self.assertIn('loss', history)
        self.assertIn('params', history)
        self.assertIn('confidence_intervals', history)
        self.assertEqual(len(history['loss']), 3)
        self.assertEqual(len(history['params']), 3)
        self.assertEqual(len(history['confidence_intervals']), 3)
    
    @unittest.skipIf(not matplotlib_available, "Matplotlib not available")
    def test_plot_calibration(self):
        """Test plotting calibration results."""
        # Run calibration first
        self.calibrator_gradient.calibrate(verbose=False)
        
        # Try plotting
        fig, axes = self.calibrator_gradient.plot_calibration()
        
        # Check that we got matplotlib objects
        self.assertIsNotNone(fig)
        self.assertEqual(len(axes), 2)


if __name__ == '__main__':
    unittest.main() 