"""
Unit tests for the jaxabm.model module.
"""
import unittest
import jax
import jax.numpy as jnp
from typing import Dict, Any
from jax import random

from jaxabm.model import Model
from jaxabm.core import ModelConfig
from jaxabm.agent import AgentCollection, AgentType

from tests.unit.test_agent import TestAgent


def simple_update_fn(env_state: Dict[str, Any], agent_states: Dict[str, Dict[str, Any]], 
                     params: Dict[str, Any], key: jax.Array) -> Dict[str, Any]:
    """Simple update function for testing."""
    # Just return the current state with a counter incremented
    new_env_state = dict(env_state) # Copy to avoid modifying original
    new_env_state['counter'] = env_state.get('counter', 0) + 1
    return new_env_state


def simple_metrics_fn(env_state: Dict[str, Any], agent_states: Dict[str, Dict[str, Any]], 
                      params: Dict[str, Any]) -> Dict[str, Any]:
    """Simple metrics function for testing."""
    # Calculate some metrics
    metrics = {}
    
    # Total value across all agents
    consumer_states = agent_states.get('consumers')
    if consumer_states and 'value' in consumer_states:
        metrics['total_value'] = jnp.sum(consumer_states['value'])
    
    # Add the counter from env_state
    metrics['step_counter'] = env_state.get('counter', 0)
    
    return metrics


class DummyAgent(AgentType):
    def init_state(self, model_config: ModelConfig, key: jax.Array) -> Dict[str, Any]:
        return {'value': jax.random.uniform(key, minval=0.0, maxval=10.0)}
    def update(self, state: Dict[str, Any], model_state: Dict[str, Any], 
               model_config: ModelConfig, key: jax.Array) -> Dict[str, Any]:
        return {'value': state['value'] + model_state['env'].get('increment', 1.0)}


class TestModel(unittest.TestCase):
    """Tests for the Model class."""
    
    def setUp(self):
        """Set up test environment."""
        self.key = jax.random.PRNGKey(0)
        self.agent_type = DummyAgent()
        self.num_agents = 10
        
        # Create agent collection placeholder (init happens in model.initialize)
        self.consumers = AgentCollection(
            agent_type=self.agent_type,
            num_agents=self.num_agents
        )
        
        # Define initial environment state
        self.initial_env_state = {
            'price_level': 1.0,
            'interest_rate': 0.05,
            'increment': 1.0 # Added for DummyAgent update
        }
        
        # Create model parameters
        self.params = {
            'adjustment_rate': 0.1,
            'target_price': 1.2
        }
        
        # Create a model config
        self.config = ModelConfig(steps=20, seed=42)
    
    def test_init(self):
        """Test initialization of Model."""
        model = Model(
            params=self.params,
            config=self.config,
            update_state_fn=simple_update_fn,
            metrics_fn=simple_metrics_fn
        )
        # Add agents and state after creation
        model.add_agent_collection('consumers', self.consumers)
        for name, value in self.initial_env_state.items():
            model.add_env_state(name, value)
        
        # Check attributes
        self.assertEqual(model._agent_collections, {'consumers': self.consumers})
        # Note: env state might include added params internally depending on impl.
        # Check initial env state variables are present
        for k, v in self.initial_env_state.items():
             self.assertEqual(model._env_state[k], v)
        self.assertEqual(model._params, self.params)
        self.assertEqual(model._update_state_fn, simple_update_fn)
        self.assertEqual(model._metrics_fn, simple_metrics_fn)
        self.assertEqual(model._time_step, 0)
        self.assertEqual(model._history, [])
        self.assertFalse(model._is_initialized)
    
    def test_add_agent_collection(self):
        """Test adding an agent collection."""
        model = Model(config=self.config)
        
        # Add a collection
        model.add_agent_collection('consumers', self.consumers)
        
        # Check that it was added
        self.assertEqual(model._agent_collections, {'consumers': self.consumers})
        
        # Should raise if model is already initialized
        model.initialize() # Call initialize instead of setting flag
        with self.assertRaises(RuntimeError):
            model.add_agent_collection('producers', self.consumers)
    
    def test_add_env_state(self):
        """Test adding environmental state variables."""
        model = Model(config=self.config)
        
        # Add state variables
        model.add_env_state('price_level', 1.0)
        model.add_env_state('interest_rate', 0.05)
        
        # Check they were added
        self.assertEqual(model._env_state, {
            'price_level': 1.0,
            'interest_rate': 0.05
        })
        
        # Add agents before initializing
        model.add_agent_collection('consumers', self.consumers)
        model.initialize() # Initialize the model
        
        # Now we can still add environment state (this behavior has changed)
        model.add_env_state('new_variable', 42.0)
        
        # Check that the state was updated in the model's state
        self.assertEqual(model.state['env']['new_variable'], 42.0)
    
    def test_model_state(self):
        """Test getting the full model state."""
        model = Model(config=self.config)
        model.add_agent_collection('consumers', self.consumers)
        for name, value in self.initial_env_state.items():
            model.add_env_state(name, value)
        
        # Initialize the model (which initializes collections)
        model.initialize()
        
        # Get the model state
        state = model.model_state()
        
        # Check structure
        self.assertIn('time_step', state)
        self.assertIn('env', state)
        self.assertIn('agents_consumers', state)
        
        # Check contents
        self.assertEqual(state['time_step'], 0)
        # Env state contains initial + potentially others
        for k, v in self.initial_env_state.items():
            self.assertEqual(state['env'][k], v)
        self.assertEqual(state['agents_consumers'], self.consumers.states)
    
    def test_initialize(self):
        """Test initializing the model."""
        model = Model(config=self.config)
        model.add_agent_collection('consumers', self.consumers)
        # Add env state needed by agent init if any (none for DummyAgent)
        
        # Initialize
        model.initialize()
        
        # Check that the model is initialized
        self.assertTrue(model._is_initialized)
        
        # Check that the agent collection is initialized
        self.assertIsNotNone(self.consumers._states)
        self.assertIsNotNone(self.consumers._key)
        self.assertIsNotNone(self.consumers.model_config)
        
        # Should raise if no agent collections
        model_no_agents = Model(config=self.config)
        with self.assertRaises(ValueError):
            model_no_agents.initialize()
    
    def test_step(self):
        """Test stepping the model."""
        model = Model(
            params=self.params,
            config=self.config,
            update_state_fn=simple_update_fn,
            metrics_fn=simple_metrics_fn
        )
        model.add_agent_collection('consumers', self.consumers)
        for name, value in self.initial_env_state.items():
            model.add_env_state(name, value)

        # Should raise if not initialized
        with self.assertRaises(RuntimeError):
            model.step()
        
        # Initialize
        model.initialize()
        initial_total_value = jnp.sum(model.agent_collections['consumers'].states['value'])
        
        # Step
        metrics = model.step()
        
        # Check that time step was incremented
        self.assertEqual(model._time_step, 1)
        
        # Check that metrics were calculated based on state *after* update
        self.assertIn('total_value', metrics)
        self.assertIn('step_counter', metrics)
        # Check counter value from metrics (reflects state *after* update_state_fn)
        self.assertEqual(metrics['step_counter'], 1)

        # Check that agent state was updated
        final_total_value = jnp.sum(model.agent_collections['consumers'].states['value'])
        expected_increase = self.num_agents * self.initial_env_state['increment']
        # Use a lower precision for floating point comparison
        self.assertAlmostEqual(final_total_value.item(), (initial_total_value + expected_increase).item(), places=4)
        
        # Check that env state was updated
        self.assertIn('counter', model._env_state)
        self.assertEqual(model._env_state['counter'], 1)
        
        # Step again and check counter in metrics and state
        metrics_step2 = model.step()
        self.assertEqual(model._env_state['counter'], 2)
        self.assertEqual(metrics_step2['step_counter'], 2)
    
    def test_run(self):
        """Test running the model."""
        model = Model(
            params=self.params,
            config=self.config, # Config now only holds seed, steps passed to run()
            update_state_fn=simple_update_fn,
            metrics_fn=simple_metrics_fn
        )
        model.add_agent_collection('consumers', self.consumers)
        for name, value in self.initial_env_state.items():
            model.add_env_state(name, value)
        
        # Run - this should initialize automatically
        num_run_steps = 15
        results = model.run(steps=num_run_steps)
        
        # Check that we have history tracked (if config enabled it - default is True)
        if self.config.track_history:
             self.assertEqual(len(model._history), num_run_steps)
        
        # Check that results contain metrics dictionary
        self.assertIsInstance(results, dict)
        self.assertIn('step', results)
        self.assertIn('total_value', results)
        self.assertIn('step_counter', results)
        
        # Check metrics dimensions match run steps
        self.assertEqual(len(results['step']), num_run_steps)
        self.assertEqual(len(results['total_value']), num_run_steps)
        # Step counter metric should go from 1 to num_run_steps
        self.assertEqual(results['step_counter'][0], 1)
        self.assertEqual(results['step_counter'][-1], num_run_steps)
        
        # Run with default steps from config
        results_default = model.run() 
        self.assertEqual(len(results_default['step']), self.config.steps)
    
    def test_agent_collections_property(self):
        """Test the agent_collections property."""
        model = Model(config=self.config)
        model.add_agent_collection('consumers', self.consumers)
        self.assertEqual(model.agent_collections, {'consumers': self.consumers})
    
    def test_state_property(self):
        """Test the state property."""
        model = Model(config=self.config)
        model.add_env_state('price_level', 1.0)
        self.assertEqual(model.state, {'env': {'price_level': 1.0}})


if __name__ == '__main__':
    unittest.main() 