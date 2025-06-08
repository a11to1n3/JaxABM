"""
Test suite for jaxabm.api module.

This module tests the AgentPy-like API interface provided by JaxABM,
including Agent, AgentList, Environment, Model, and Results classes.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from unittest.mock import Mock, patch
from typing import Dict, Any

from jaxabm.api import (
    Agent, AgentTypeWrapper, AgentList, Environment, 
    Results, Model
)
from jaxabm.core import ModelConfig


class TestAgent:
    """Test cases for the Agent class."""
    
    def test_agent_initialization(self):
        """Test basic agent initialization."""
        agent = Agent()
        assert agent.id is None
        assert agent.model is None
        assert agent.p == {}
        assert agent._state == {}
    
    def test_agent_setup_default(self):
        """Test default setup method returns empty dict."""
        agent = Agent()
        result = agent.setup()
        assert result == {}
    
    def test_agent_step_default(self):
        """Test default step method returns current state."""
        agent = Agent()
        agent._state = {"x": 1, "y": 2}
        result = agent.step()
        assert result == {"x": 1, "y": 2}
    
    def test_agent_step_with_model_state(self):
        """Test step method with model state parameter."""
        agent = Agent()
        agent._state = {"value": 10}
        model_state = {"global_param": 5}
        result = agent.step(model_state)
        assert result == {"value": 10}


class TestCustomAgent(Agent):
    """Custom agent class for testing inheritance."""
    
    def setup(self):
        return {"x": 0, "y": 0, "energy": 100}
    
    def step(self, model_state=None):
        current_state = self._state.copy()
        current_state["x"] += 1
        current_state["y"] += 1
        current_state["energy"] -= 1
        return current_state


class TestAgentTypeWrapper:
    """Test cases for AgentTypeWrapper class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent_class = TestCustomAgent
        self.params = {"max_energy": 100, "speed": 1}
        self.config = ModelConfig(seed=42)
        self.key = jax.random.PRNGKey(42)
    
    def test_wrapper_initialization(self):
        """Test AgentTypeWrapper initialization."""
        wrapper = AgentTypeWrapper(self.agent_class, self.params)
        assert wrapper.agent_class == self.agent_class
        assert wrapper.params == self.params
        assert isinstance(wrapper.agent_instance, TestCustomAgent)
        assert wrapper.agent_instance.p == self.params
    
    def test_wrapper_initialization_no_params(self):
        """Test AgentTypeWrapper initialization without parameters."""
        wrapper = AgentTypeWrapper(self.agent_class)
        assert wrapper.params == {}
        assert wrapper.agent_instance.p == {}
    
    def test_init_state(self):
        """Test init_state method."""
        wrapper = AgentTypeWrapper(self.agent_class, self.params)
        state = wrapper.init_state(self.config, self.key)
        expected_state = {"x": 0, "y": 0, "energy": 100}
        assert state == expected_state
    
    def test_init_state_invalid_return(self):
        """Test init_state with invalid return type from setup."""
        class BadAgent(Agent):
            def setup(self):
                return "invalid"
        
        wrapper = AgentTypeWrapper(BadAgent)
        with pytest.raises(ValueError, match="Agent.setup\\(\\) must return a dictionary"):
            wrapper.init_state(self.config, self.key)
    
    def test_init_state_none_return(self):
        """Test init_state with None return from setup."""
        class NoneAgent(Agent):
            def setup(self):
                return None
        
        wrapper = AgentTypeWrapper(NoneAgent)
        state = wrapper.init_state(self.config, self.key)
        assert state == {}
    
    def test_update(self):
        """Test update method."""
        wrapper = AgentTypeWrapper(self.agent_class, self.params)
        initial_state = {"x": 5, "y": 10, "energy": 50}
        model_state = {"global_param": 42}
        
        new_state = wrapper.update(initial_state, model_state, self.config, self.key)
        expected_state = {"x": 6, "y": 11, "energy": 49}
        assert new_state == expected_state
    
    def test_update_invalid_return(self):
        """Test update with invalid return type from step."""
        class BadStepAgent(Agent):
            def step(self, model_state=None):
                return "invalid"
        
        wrapper = AgentTypeWrapper(BadStepAgent)
        state = {"x": 1}
        with pytest.raises(ValueError, match="Agent.step\\(\\) must return a dictionary"):
            wrapper.update(state, {}, self.config, self.key)
    
    def test_update_none_return(self):
        """Test update with None return from step."""
        class NoneStepAgent(Agent):
            def step(self, model_state=None):
                return None
        
        wrapper = AgentTypeWrapper(NoneStepAgent)
        initial_state = {"x": 1}
        new_state = wrapper.update(initial_state, {}, self.config, self.key)
        assert new_state == initial_state


class TestAgentList:
    """Test cases for AgentList class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = Mock()
        self.model._model = Mock()
        self.model._model.state = {}
        self.agent_class = TestCustomAgent
        self.n = 5
        self.params = {"speed": 2}
    
    def test_agent_list_initialization(self):
        """Test AgentList initialization."""
        agent_list = AgentList(self.model, self.n, self.agent_class, **self.params)
        
        assert agent_list.model == self.model
        assert agent_list.n == self.n
        assert agent_list.agent_class == self.agent_class
        assert agent_list.params == self.params
        assert isinstance(agent_list.agent_type, AgentTypeWrapper)
        assert agent_list.collection.num_agents == self.n
        assert agent_list.name is None
    
    def test_agent_list_states_no_name(self):
        """Test states property when agent list has no name."""
        agent_list = AgentList(self.model, self.n, self.agent_class)
        states = agent_list.states
        assert states == {}
    
    def test_agent_list_states_with_name(self):
        """Test states property when agent list has a name."""
        agent_list = AgentList(self.model, self.n, self.agent_class)
        agent_list.name = "test_agents"
        
        # Mock model state with agent states
        mock_agent_states = {
            "test_agents": {
                "x": [1, 2, 3, 4, 5],
                "y": [2, 3, 4, 5, 6]
            }
        }
        self.model._model.state = {"agents": mock_agent_states}
        
        states = agent_list.states
        assert states == mock_agent_states["test_agents"]
    
    def test_agent_list_getattr(self):
        """Test __getattr__ method for accessing agent properties."""
        agent_list = AgentList(self.model, self.n, self.agent_class)
        agent_list.name = "test_agents"
        
        # Mock model state with agent states
        mock_agent_states = {
            "test_agents": {
                "x": [1, 2, 3, 4, 5],
                "energy": [90, 95, 88, 92, 87]
            }
        }
        self.model._model.state = {"agents": mock_agent_states}
        
        # Test accessing properties
        assert agent_list.x == [1, 2, 3, 4, 5]
        assert agent_list.energy == [90, 95, 88, 92, 87]
    
    def test_agent_list_getattr_no_property(self):
        """Test __getattr__ when property doesn't exist."""
        agent_list = AgentList(self.model, self.n, self.agent_class)
        agent_list.name = "test_agents"
        
        self.model._model.state = {"agents": {"test_agents": {}}}
        
        with pytest.raises(AttributeError):
            _ = agent_list.nonexistent_property
    
    def test_agent_list_len(self):
        """Test __len__ method."""
        agent_list = AgentList(self.model, self.n, self.agent_class)
        assert len(agent_list) == self.n


class TestEnvironment:
    """Test cases for Environment class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = Mock()
        self.model._model = None
    
    def test_environment_initialization(self):
        """Test Environment initialization."""
        env = Environment(self.model)
        assert env.model == self.model
    
    def test_add_state(self):
        """Test add_state method."""
        env = Environment(self.model)
        # Remove the _model attribute to avoid the add_env_state call
        if hasattr(self.model, '_model'):
            delattr(self.model, '_model')
        
        env.add_state("temperature", 25.0)
        assert env.state["temperature"] == 25.0
        
        env.add_state("humidity", 60)
        assert env.state["humidity"] == 60
    
    def test_getattr(self):
        """Test __getattr__ method for accessing environment state."""
        env = Environment(self.model)
        env.state = {"temperature": 25.0, "pressure": 1013.25}
        
        assert env.temperature == 25.0
        assert env.pressure == 1013.25
    
    def test_getattr_nonexistent(self):
        """Test __getattr__ for nonexistent attributes."""
        env = Environment(self.model)
        env.state = {}
        
        with pytest.raises(AttributeError):
            _ = env.nonexistent_property


class TestResults:
    """Test cases for Results class."""
    
    def test_results_initialization(self):
        """Test Results initialization."""
        data = {
            "step": [0, 1, 2, 3],
            "metric1": [10, 15, 20, 25],
            "metric2": [5, 8, 12, 18]
        }
        results = Results(data)
        assert results._data == data
    
    def test_results_agents_property(self):
        """Test agents property via variables property."""
        data = {
            "agents.type1.x": [[1, 2], [3, 4], [5, 6]],
            "agents.type1.y": [[2, 3], [4, 5], [6, 7]]
        }
        results = Results(data)
        variables = results.variables
        assert hasattr(variables, 'type1')
    
    def test_results_variables_property(self):
        """Test variables property."""
        data = {
            "temp": [20, 21, 22],
            "count": [100, 105, 110]
        }
        results = Results(data)
        variables = results.variables
        assert variables._data == data
    
    def test_results_plot_basic(self):
        """Test basic plot functionality."""
        data = {
            "step": [0, 1, 2, 3],
            "metric1": [10, 15, 20, 25]
        }
        results = Results(data)
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig, mock_ax = Mock(), Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            results.plot(['metric1'])
            mock_subplots.assert_called_once()
            mock_ax.plot.assert_called_once_with([10, 15, 20, 25], label='metric1')
    
    def test_variable_container(self):
        """Test VariableContainer class."""
        data = {"temp": [20, 21, 22], "pressure": [1013, 1014, 1015]}
        container = Results.VariableContainer(data)
        assert container._data == data
    
    def test_variable_series(self):
        """Test VariableSeries class."""
        values = [10, 15, 20, 25, 30]
        series = Results.VariableContainer.AgentContainer.VariableSeries(values, "test_metric")
        
        assert series._values == values
        assert series._name == "test_metric"
        assert len(series) == 5
        assert series[0] == 10
        assert series[-1] == 30
    
    def test_variable_series_plot(self):
        """Test VariableSeries plot method."""
        values = [10, 15, 20, 25, 30]
        series = Results.VariableContainer.AgentContainer.VariableSeries(values, "test_metric")
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig, mock_ax = Mock(), Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            result_ax = series.plot()
            
            mock_subplots.assert_called_once()
            mock_ax.plot.assert_called_once_with(values)
            mock_ax.set_xlabel.assert_called_once_with('Time')
            mock_ax.set_ylabel.assert_called_once_with('test_metric')
            assert result_ax == mock_ax


class TestModel:
    """Test cases for Model class."""
    
    def test_model_initialization_defaults(self):
        """Test Model initialization with default parameters."""
        model = Model()
        assert model.p == {}
        assert model.seed is not None
        assert isinstance(model.seed, int)
        assert model._agent_lists == {}
        assert model._recorded_data == {}
        assert model._model is None
    
    def test_model_initialization_with_params(self):
        """Test Model initialization with parameters."""
        params = {"param1": 10, "param2": "test"}
        model = Model(parameters=params, seed=42)
        assert model.p == params
        assert model.seed == 42
    
    def test_model_setup(self):
        """Test setup method (should be overridden)."""
        model = Model()
        # Default setup does nothing
        model.setup()
        # No assertion needed as setup is a no-op by default
    
    def test_model_step(self):
        """Test step method (should be overridden)."""
        model = Model()
        # Default step does nothing
        model.step()
        # No assertion needed as step is a no-op by default
    
    def test_model_update_state(self):
        """Test update_state method."""
        model = Model()
        env_state = {"temperature": 25}
        agent_states = {"agents1": {"x": [1, 2, 3]}}
        model_params = {"param1": 10}
        key = jax.random.PRNGKey(42)
        
        result = model.update_state(env_state, agent_states, model_params, key)
        
        # Default implementation returns environment state
        assert result == env_state
    
    def test_model_compute_metrics(self):
        """Test compute_metrics method."""
        model = Model()
        env_state = {"temperature": 25}
        agent_states = {"agents1": {"x": [1, 2, 3]}}
        model_params = {"param1": 10}
        
        result = model.compute_metrics(env_state, agent_states, model_params)
        
        # Default implementation returns empty dict
        assert result == {}
    
    def test_model_add_agents(self):
        """Test add_agents method."""
        model = Model()
        agent_list = model.add_agents(5, TestCustomAgent, name="test_agents", speed=2)
        
        assert isinstance(agent_list, AgentList)
        assert agent_list.n == 5
        assert agent_list.agent_class == TestCustomAgent
        assert agent_list.name == "test_agents"
        assert agent_list.params == {"speed": 2}
        assert "test_agents" in model._agent_lists
    
    def test_model_add_agents_auto_name(self):
        """Test add_agents with automatic naming."""
        model = Model()
        agent_list = model.add_agents(3, TestCustomAgent)
        
        expected_name = "testcustomagents"  # lowercase + 's'
        assert agent_list.name == expected_name
        assert expected_name in model._agent_lists
    
    def test_model_record(self):
        """Test record method."""
        model = Model()
        model.record("metric1", 42)
        model.record("metric2", [1, 2, 3])
        
        assert model._recorded_data["metric1"] == [42]
        assert model._recorded_data["metric2"] == [[1, 2, 3]]
        
        # Record again to test list accumulation
        model.record("metric1", 43)
        assert model._recorded_data["metric1"] == [42, 43]
    
    @patch('jaxabm.api.JaxModel')
    def test_model_run_basic(self, mock_jax_model_class):
        """Test basic run method."""
        # Mock the JaxModel and its behavior
        mock_model_instance = Mock()
        mock_jax_model_class.return_value = mock_model_instance
        
        # Mock the run method to return some results
        mock_results = {
            "step": [0, 1, 2],
            "metric1": [10, 15, 20]
        }
        mock_model_instance.run.return_value = mock_results
        mock_model_instance.state = {}  # Mock empty state
        
        model = Model()
        model.setup = Mock()  # Mock setup method
        
        results = model.run(steps=3)
        
        # Verify setup was called
        model.setup.assert_called_once()
        
        # Verify JaxModel was created and run
        mock_jax_model_class.assert_called_once()
        mock_model_instance.run.assert_called_once()
        
        # Verify Results object was created
        assert isinstance(results, Results)
    
    @patch('jaxabm.api.JaxModel')
    def test_model_run_with_recorded_data(self, mock_jax_model_class):
        """Test run method with recorded data."""
        mock_model_instance = Mock()
        mock_jax_model_class.return_value = mock_model_instance
        mock_model_instance.run.return_value = {}
        mock_model_instance.state = {}  # Mock empty state
        
        model = Model()
        model.setup = Mock()
        model._recorded_data = {"custom_metric": [1, 2, 3]}
        
        results = model.run(steps=3)
        
        # Should merge recorded data with model results
        assert isinstance(results, Results)
        assert "custom_metric" in results._data
    
    def test_model_environment_property(self):
        """Test environment property."""
        model = Model()
        env = model.env
        assert isinstance(env, Environment)
        assert env.model == model


# Integration tests
class TestAPIIntegration:
    """Integration tests for the API components."""
    
    def test_complete_model_workflow(self):
        """Test complete workflow with custom model and agents."""
        
        class TestModel(Model):
            def setup(self):
                self.agents = self.add_agents(3, TestCustomAgent, name="test_agents")
                # Don't add state to env during setup since _model is None
                
            def step(self):
                # Record some metrics
                self.record("temperature", 20.0)
                self.record("step_count", len(self._recorded_data.get("temperature", [])))
        
        # This test mainly checks that the API doesn't crash
        # The actual JAX model execution is mocked in other tests
        model = TestModel(parameters={"test_param": 42}, seed=123)
        
        # Test initialization
        assert model.p["test_param"] == 42
        assert model.seed == 123
        
        # Test setup
        model.setup()
        assert "test_agents" in model._agent_lists
        
        # Test step
        model.step()
        assert "temperature" in model._recorded_data
        assert "step_count" in model._recorded_data 