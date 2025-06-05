"""Tests for Model.jit_step functionality."""
import unittest
from typing import Dict, Any

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    HAS_JAX = True
except ImportError:  # pragma: no cover - environment without jax
    HAS_JAX = False

if HAS_JAX:
    from jaxabm.model import Model
    from jaxabm.core import ModelConfig
    from jaxabm.agent import AgentCollection, AgentType


    def simple_update_fn(env_state: Dict[str, Any], agent_states: Dict[str, Dict[str, Any]],
                         params: Dict[str, Any], key: jax.Array) -> Dict[str, Any]:
        """Increment a counter in the environment state."""
        new_env_state = dict(env_state)
        new_env_state['counter'] = env_state.get('counter', 0) + 1
        return new_env_state


    def simple_metrics_fn(env_state: Dict[str, Any], agent_states: Dict[str, Dict[str, Any]],
                          params: Dict[str, Any]) -> Dict[str, Any]:
        """Return the current counter value as a metric."""
        return {'counter': env_state.get('counter', 0)}


    class PositionAgent(AgentType):
        """Minimal agent holding a position value."""

        def init_state(self, model_config: ModelConfig, key: jax.Array) -> Dict[str, Any]:
            return {'position': jnp.array(0.0)}

        def update(self, state: Dict[str, Any], model_state: Dict[str, Any],
                   model_config: ModelConfig, key: jax.Array) -> Dict[str, Any]:
            return {'position': state['position'] + 1.0}


class TestModelJit(unittest.TestCase):
    """Tests for the jit_step method of Model."""

    @unittest.skipIf(not HAS_JAX, "JAX not installed")
    def test_jit_step_matches_step(self):
        num_agents = 5
        params: Dict[str, Any] = {}
        config = ModelConfig(seed=0, steps=1)

        # Regular model to obtain baseline results
        model_step = Model(params=params, config=config,
                           update_state_fn=simple_update_fn,
                           metrics_fn=simple_metrics_fn)
        model_step.add_agent_collection('agents', AgentCollection(PositionAgent(), num_agents))
        model_step.add_env_state('counter', 0)
        model_step.initialize()

        metrics_step = model_step.step()
        env_after_step = dict(model_step._env_state)

        # Separate model used with jit_step
        model_jit = Model(params=params, config=config,
                          update_state_fn=simple_update_fn,
                          metrics_fn=simple_metrics_fn)
        model_jit.add_agent_collection('agents', AgentCollection(PositionAgent(), num_agents))
        model_jit.add_env_state('counter', 0)
        model_jit.initialize()

        jit_fn = model_jit.jit_step()
        env_state = model_jit._env_state
        agent_states = {name: coll.states for name, coll in model_jit.agent_collections.items()}
        key = model_jit._rng
        env_jit, agent_states_jit, metrics_jit, _ = jit_fn(env_state, agent_states, params, key)

        self.assertEqual(int(env_jit['counter']), env_after_step['counter'])
        self.assertEqual(int(metrics_jit['counter']), metrics_step['counter'])


if __name__ == '__main__':
    unittest.main()
