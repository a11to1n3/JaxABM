"""
Pytest configuration for the JaxABM test suite.

This file contains fixtures and configuration for pytest.
"""
import pytest
import sys
import warnings

# Try to import JAX, but handle the case where it's not installed
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    warnings.warn("JAX is not installed. Tests requiring JAX will be skipped.")
    HAS_JAX = False
    # Create dummy jax module
    class DummyJax:
        def __getattr__(self, name):
            return None
    jax = DummyJax()
    jnp = None

from typing import Dict, Any

# Only import these if we have JAX
if HAS_JAX:
    from jaxabm.agent import AgentType, AgentCollection
    from jaxabm.core import ModelConfig
    from jaxabm.model import Model

# Skip tests requiring JAX if it's not installed
jax_required = pytest.mark.skipif(not HAS_JAX, reason="JAX is not installed")


# This fixture ensures deterministic test execution
@pytest.fixture(scope="function")
def random_seed():
    """Fixed random seed for deterministic tests."""
    return 42


# These fixtures require JAX
if HAS_JAX:
    @pytest.fixture(scope="function")
    def random_key(random_seed):
        """Fixed JAX PRNG key for deterministic tests."""
        return jax.random.PRNGKey(random_seed)


    # Simple agent type for testing
    class SimpleAgent(AgentType):
        """Minimal agent implementation for testing."""
        
        growth_rate: float = 0.1

        def init_state(self, model_config: ModelConfig, key: jax.Array) -> Dict[str, Any]:
            """Initialize agent state."""
            return {
                'value': jnp.array(0.0),
                'id': jnp.array(jax.random.randint(key, (), 0, 1000000))
            }
        
        def update(self, state: Dict[str, Any], model_state: Dict[str, Any], 
                   model_config: ModelConfig, key: jax.Array) -> Dict[str, Any]:
            """Update agent state."""
            new_state = {
                'value': state['value'] * (1.0 + self.growth_rate),
                'id': state['id']
            }
            return new_state


    @pytest.fixture(scope="function")
    def simple_agent_type():
        """Fixture providing a simple agent type for testing."""
        return SimpleAgent


    @pytest.fixture(scope="function")
    def simple_agent_collection(simple_agent_type, random_key):
        """Fixture providing a simple agent collection."""
        collection = AgentCollection(
            agent_type=simple_agent_type(),
            num_agents=10
        )
        return collection


    def simple_update_fn(env_state: Dict[str, Any], agent_states: Dict[str, Dict[str, Any]], 
                         params: Dict[str, Any], key: jax.Array) -> Dict[str, Any]:
        """Simple update function for testing."""
        new_env_state = dict(env_state)
        new_env_state['counter'] = env_state.get('counter', 0) + 1
        return new_env_state


    def simple_metrics_fn(env_state: Dict[str, Any], agent_states: Dict[str, Dict[str, Any]], 
                          params: Dict[str, Any]) -> Dict[str, Any]:
        """Simple metrics function for testing."""
        metrics = {}
        metrics['counter'] = env_state.get('counter', 0)
        
        agent_state_data = agent_states.get('agents')
        if agent_state_data and 'value' in agent_state_data:
            metrics['total_value'] = jnp.sum(agent_state_data['value'])
        
        return metrics


    @pytest.fixture(scope="function")
    def simple_model(simple_agent_collection, random_seed):
        """Fixture providing a simple model for testing."""
        model_params = {'some_model_param': 1.0}
        initial_env_state = {'counter': 0}
        config = ModelConfig(seed=random_seed)

        model = Model(
            params=model_params,
            config=config,
            update_state_fn=simple_update_fn,
            metrics_fn=simple_metrics_fn,
        )
        model.add_agent_collection('agents', simple_agent_collection)
        for name, value in initial_env_state.items():
            model.add_env_state(name, value)
            
        return model 