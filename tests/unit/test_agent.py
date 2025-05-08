"""
Tests for the agent module of the AgentJax framework.

This module tests the functionality of agent classes, including initialization,
state updates, and collection operations.
"""

import unittest
import numpy as np
from typing import Dict, Any, List, Tuple, Protocol, Callable

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

if HAS_JAX:
    from jaxabm.agent import AgentType, AgentCollection
    from jaxabm.core import ModelConfig
else:
    # Create dummy classes for type hints when JAX is not available
    class AgentType(Protocol):
        """Protocol for agent types in legacy mode."""
        def init_state(self, model_config: Any, key: Any) -> Dict[str, Any]:
            ...
        def update(self, state: Dict[str, Any], model_state: Dict[str, Any], 
                  model_config: Any, key: Any) -> Dict[str, Any]:
            ...

    class AgentCollection:
        """Dummy AgentCollection for type hints when JAX is not available."""
        def __init__(self, *args, **kwargs):
            pass
        
    class ModelConfig:
        """Dummy ModelConfig for type hints when JAX is not available."""
        def __init__(self, *args, **kwargs):
            pass

# Define test agents for unit testing
class TestAgent:
    """Test agent class."""
    
    def init_state(self, model_config: ModelConfig, key: Any) -> Dict[str, Any]:
        """Initialize agent state.
        
        Args:
            model_config: The model configuration
            key: Random key
            
        Returns:
            Initial agent state
        """
        if HAS_JAX:
            key1, key2 = random.split(key)
            return {
                'wealth': random.uniform(key1, minval=0.0, maxval=100.0),
                'productivity': random.uniform(key2, minval=0.5, maxval=1.5)
            }
        else:
            np.random.seed(0)
            return {
                'wealth': np.random.uniform(0.0, 100.0),
                'productivity': np.random.uniform(0.5, 1.5)
            }
    
    def update(self, state: Dict[str, Any], model_state: Dict[str, Any], 
              model_config: ModelConfig, key: Any) -> Dict[str, Any]:
        """Update agent state.
        
        Args:
            state: Current agent state
            model_state: Current model state
            model_config: Model configuration
            key: Random key
            
        Returns:
            Updated agent state
        """
        if HAS_JAX:
            # Simple update logic for testing
            income = state['productivity'] * model_state.get('wage_rate', 1.0)
            new_wealth = state['wealth'] + income
            
            return {
                'wealth': new_wealth,
                'productivity': state['productivity']
            }
        else:
            # Simple update logic for testing
            income = state['productivity'] * model_state.get('wage_rate', 1.0)
            new_wealth = state['wealth'] + income
            
            return {
                'wealth': new_wealth,
                'productivity': state['productivity']
            }

class LegacyTestAgent:
    """Legacy implementation of the test agent for comparison."""
    
    def __init__(self, agent_id):
        """Initialize agent with ID."""
        self.agent_id = agent_id
        np.random.seed(agent_id)  # For reproducibility
        self.wealth = np.random.uniform(0.0, 100.0)
        self.productivity = np.random.uniform(0.5, 1.5)
    
    def step(self, model_state):
        """Update agent state based on model state."""
        income = self.productivity * model_state.get('wage_rate', 1.0)
        self.wealth += income
        return {'wealth': self.wealth, 'productivity': self.productivity}


class TestModelConfig(unittest.TestCase):
    """Test suite for ModelConfig class."""
    
    @unittest.skipIf(not HAS_JAX, "JAX not installed")
    def test_model_config_init(self):
        """Test ModelConfig initialization."""
        config = ModelConfig(
            seed=42,
            steps=10,
            track_history=True,
            collect_interval=2
        )
        
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.steps, 10)
        self.assertEqual(config.track_history, True)
        self.assertEqual(config.collect_interval, 2)


class TestAgentCollection(unittest.TestCase):
    """Test suite for AgentCollection class."""
    
    @unittest.skipIf(not HAS_JAX, "JAX not installed")
    def test_init_agents(self):
        """Test agent initialization."""
        key = random.PRNGKey(0)
        model_config = ModelConfig(
            seed=42,
            steps=5
        )
        
        agent_type = TestAgent()
        agents = AgentCollection(agent_type, 10)
        
        # Initialize the agent collection manually
        agents.init(key, model_config)
        
        # Check that agents were initialized
        self.assertEqual(agents.num_agents, 10)
        
        # Check that agent states have the right variables
        agent_states = agents.get_states()
        self.assertIn('wealth', agent_states)
        self.assertIn('productivity', agent_states)
        
        # Check shapes
        self.assertEqual(agent_states['wealth'].shape, (10,))
        self.assertEqual(agent_states['productivity'].shape, (10,))
        
        # Check that values are within expected ranges
        self.assertTrue(jnp.all(agent_states['wealth'] >= 0.0))
        self.assertTrue(jnp.all(agent_states['wealth'] <= 100.0))
        self.assertTrue(jnp.all(agent_states['productivity'] >= 0.5))
        self.assertTrue(jnp.all(agent_states['productivity'] <= 1.5))
    
    @unittest.skipIf(not HAS_JAX, "JAX not installed")
    def test_update_agents(self):
        """Test agent state updates."""
        key = random.PRNGKey(0)
        model_config = ModelConfig(
            seed=42,
            steps=5
        )
        
        agent_type = TestAgent()
        agents = AgentCollection(agent_type, 10)
        
        # Initialize the agent collection manually
        agents.init(key, model_config)
        
        # Get initial states
        initial_states = agents.get_states()
        initial_wealth = initial_states['wealth']
        
        # Update agents
        model_state = {'wage_rate': 1.0}
        new_key = random.PRNGKey(1)
        
        # The update method modifies internal state and returns nothing
        agents.update(model_state, new_key, model_config)
        
        # Get updated states
        updated_states = agents.get_states()
        updated_wealth = updated_states['wealth']
        
        # Check that wealth increased by productivity amount
        expected_increase = initial_states['productivity']
        actual_increase = updated_wealth - initial_wealth
        
        # Allow for small floating point differences
        self.assertTrue(jnp.allclose(actual_increase, expected_increase, rtol=1e-5))
    
    @unittest.skipIf(not HAS_JAX, "JAX not installed")
    def test_aggregate(self):
        """Test aggregation of agent states."""
        key = random.PRNGKey(0)
        model_config = ModelConfig(
            seed=42,
            steps=5
        )
        
        agent_type = TestAgent()
        agents = AgentCollection(agent_type, 10)
        
        # Initialize the agent collection manually
        agents.init(key, model_config)
        
        # Test sum aggregation
        total_wealth = agents.aggregate('wealth', jnp.sum)
        manual_sum = jnp.sum(agents.get_states()['wealth'])
        self.assertAlmostEqual(float(total_wealth), float(manual_sum))
        
        # Test mean aggregation
        mean_productivity = agents.aggregate('productivity', jnp.mean)
        manual_mean = jnp.mean(agents.get_states()['productivity'])
        self.assertAlmostEqual(float(mean_productivity), float(manual_mean))
    
    @unittest.skipIf(not HAS_JAX, "JAX not installed")
    def test_filter(self):
        """Test filtering of agents based on conditions."""
        key = random.PRNGKey(0)
        model_config = ModelConfig(
            seed=42,
            steps=5
        )
        
        agent_type = TestAgent()
        agents = AgentCollection(agent_type, 10)
        
        # Initialize the agent collection manually
        agents.init(key, model_config)
        
        # Filter wealthy agents (wealth > 50)
        def wealth_condition(states):
            return states['wealth'] > 50.0
        
        wealthy_agents = agents.filter(wealth_condition)
        
        # Check that all wealthy agents have wealth > 50
        wealthy_states = wealthy_agents.get_states()
        self.assertTrue(jnp.all(wealthy_states['wealth'] > 50.0))
        
        # Check that the filtered collection is smaller
        self.assertLessEqual(wealthy_agents.num_agents, agents.num_agents)


if __name__ == '__main__':
    unittest.main() 