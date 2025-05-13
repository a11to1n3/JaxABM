"""
AgentPy-like API for JaxABM.

This module provides an AgentPy-like interface for the JaxABM framework, making
it easier to create and run agent-based models while maintaining the performance
benefits of JAX acceleration.

The main classes are:
    - Agent: Base class for creating agents
    - AgentList: Container for managing collections of agents
    - Environment: Container for environment state
    - Model: Base class for creating models
    - Results: Container for simulation results
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable, TypeVar, Type
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import warnings
import time
import matplotlib.pyplot as plt

# Import core components
from .agent import AgentType, AgentCollection
from .core import ModelConfig
from .model import Model as JaxModel
from .utils import convert_to_numpy, format_time

# Type variables for better type hinting
T = TypeVar('T', bound='Agent')
ModelType = TypeVar('ModelType', bound='Model')


class Agent:
    """Base class for agents in JaxABM.
    
    This class provides an AgentPy-like interface for creating agents. To create
    a custom agent, inherit from this class and override the setup and step methods.
    
    Example:
        ```python
        class MyAgent(Agent):
            def setup(self):
                self.x = 0
                self.y = 0
                
            def step(self):
                self.x += 1
                self.y += 1
        ```
    """
    
    def __init__(self):
        """Initialize agent."""
        self.id = None
        self.model = None
        self.p = {}  # Parameters
        self._state = {}
    
    def setup(self) -> Dict[str, Any]:
        """Set up agent state.
        
        Override this method to initialize agent state.
        
        Returns:
            A dictionary containing the initial agent state.
        """
        return {}
    
    def step(self, model_state: Dict[str, Any] = None) -> Dict[str, Any]:
        """Update agent state.
        
        Override this method to define agent behavior.
        
        Args:
            model_state: The current model state (optional).
            
        Returns:
            A dictionary containing the updated agent state.
        """
        return self._state


class AgentTypeWrapper(AgentType):
    """Wrapper to adapt Agent class to AgentType protocol."""
    
    def __init__(self, agent_class, params=None):
        """Initialize agent type wrapper.
        
        Args:
            agent_class: The Agent class to wrap.
            params: Parameters to pass to the agent.
        """
        self.agent_class = agent_class
        self.params = params or {}
        
        # Create an instance to access methods
        self.agent_instance = agent_class()
        if params:
            self.agent_instance.p = params
    
    def init_state(self, model_config: ModelConfig, key: jax.Array) -> Dict[str, Any]:
        """Initialize agent state.
        
        Args:
            model_config: Model configuration.
            key: JAX random key.
            
        Returns:
            Initial agent state dictionary.
        """
        # Call the agent's setup method
        state = self.agent_instance.setup()
        
        # Ensure state is a dictionary
        if not isinstance(state, dict):
            if state is None:
                return {}
            else:
                raise ValueError(f"Agent.setup() must return a dictionary, got {type(state)}")
        
        return state
    
    def update(self, state: Dict[str, Any], model_state: Dict[str, Any],
               model_config: ModelConfig, key: jax.Array) -> Dict[str, Any]:
        """Update agent state.
        
        Args:
            state: Current agent state.
            model_state: Current model state.
            model_config: Model configuration.
            key: JAX random key.
            
        Returns:
            Updated agent state.
        """
        # Update the agent instance's state
        self.agent_instance._state = state
        
        # Call the agent's step method
        new_state = self.agent_instance.step(model_state)
        
        # Ensure new_state is a dictionary
        if not isinstance(new_state, dict):
            if new_state is None:
                return state  # No change
            else:
                raise ValueError(f"Agent.step() must return a dictionary, got {type(new_state)}")
        
        return new_state


class AgentList:
    """Container for managing collections of agents.
    
    This class provides an AgentPy-like interface for managing groups of agents.
    
    Example:
        ```python
        agents = AgentList(model, 10, MyAgent)
        agents.step()
        ```
    """
    
    def __init__(self, model: 'Model', n: int, agent_class: Type[Agent], **kwargs):
        """Initialize agent list.
        
        Args:
            model: The model the agents belong to.
            n: Number of agents to create.
            agent_class: The Agent class to use.
            **kwargs: Parameters to pass to the agents.
        """
        self.model = model
        self.n = n
        self.agent_class = agent_class
        self.params = kwargs
        
        # Create AgentTypeWrapper and AgentCollection
        self.agent_type = AgentTypeWrapper(agent_class, kwargs)
        self.collection = AgentCollection(
            agent_type=self.agent_type,
            num_agents=n
        )
        
        # Store name for use with model
        self.name = None
    
    @property
    def states(self) -> Dict[str, Any]:
        """Get the current states of all agents.
        
        Returns:
            Dictionary of agent states by property.
        """
        if hasattr(self.model, '_model') and self.model._model.state:
            agent_states = self.model._model.state.get('agents', {})
            if self.name and self.name in agent_states:
                return agent_states[self.name]
        return {}
    
    def __getattr__(self, name: str) -> Any:
        """Get agent attribute.
        
        This allows accessing agent attributes like agents.x to get an array
        of all agents' x values.
        
        Args:
            name: Attribute name.
            
        Returns:
            Array of attribute values.
        """
        states = self.states
        if name in states:
            return states[name]
        
        # Default behavior for unknown attributes
        raise AttributeError(f"'AgentList' object has no attribute '{name}'")
    
    def __len__(self) -> int:
        """Get number of agents.
        
        Returns:
            Number of agents.
        """
        return self.n


class Environment:
    """Environment for agent interactions.
    
    This class provides a container for environment state and methods.
    
    Example:
        ```python
        env = Environment(model)
        env.add_state('temperature', 25.0)
        ```
    """
    
    def __init__(self, model: 'Model'):
        """Initialize environment.
        
        Args:
            model: The model the environment belongs to.
        """
        self.model = model
        self.state = {}
    
    def add_state(self, name: str, value: Any) -> None:
        """Add state variable to environment.
        
        Args:
            name: State variable name.
            value: State variable value.
        """
        self.state[name] = value
        
        # Add to model if it exists
        if hasattr(self.model, '_model'):
            self.model._model.add_env_state(name, value)
    
    def __getattr__(self, name: str) -> Any:
        """Get environment state variable.
        
        Args:
            name: State variable name.
            
        Returns:
            State variable value.
        """
        if name in self.state:
            return self.state[name]
        
        # Check model state if environment state isn't available
        if hasattr(self.model, '_model') and self.model._model.state:
            env_state = self.model._model.state.get('env', {})
            if name in env_state:
                return env_state[name]
        
        # Default behavior for unknown attributes
        raise AttributeError(f"'Environment' object has no attribute '{name}'")


class Results:
    """Container for simulation results.
    
    This class provides an AgentPy-like interface for accessing and
    visualizing simulation results.
    
    Example:
        ```python
        results = model.run()
        results.variables.Agent.x.plot()
        ```
    """
    
    class VariableContainer:
        """Container for simulation variables."""
        
        def __init__(self, data: Dict[str, Any]):
            """Initialize variable container.
            
            Args:
                data: Simulation data.
            """
            self._data = data
            
            # Create dynamic attributes for each agent type
            agent_types = set()
            for key in data:
                if key.startswith('agents.'):
                    agent_type = key.split('.')[1]
                    agent_types.add(agent_type)
            
            # Create nested containers for agent types
            for agent_type in agent_types:
                setattr(self, agent_type, self.AgentContainer(data, agent_type))
        
        class AgentContainer:
            """Container for agent variables."""
            
            def __init__(self, data: Dict[str, Any], agent_type: str):
                """Initialize agent container.
                
                Args:
                    data: Simulation data.
                    agent_type: Agent type name.
                """
                self._data = data
                self._agent_type = agent_type
                
                # Create dynamic attributes for each agent variable
                for key in data:
                    if key.startswith(f'agents.{agent_type}.'):
                        var_name = key.split('.')[-1]
                        setattr(self, var_name, self.VariableSeries(data[key], var_name))
            
            class VariableSeries:
                """Series of variable values over time."""
                
                def __init__(self, values: List[Any], name: str):
                    """Initialize variable series.
                    
                    Args:
                        values: List of values over time.
                        name: Variable name.
                    """
                    self._values = values
                    self._name = name
                
                def plot(self, ax=None, **kwargs):
                    """Plot variable values over time.
                    
                    Args:
                        ax: Matplotlib axis.
                        **kwargs: Additional keyword arguments for plotting.
                    
                    Returns:
                        Matplotlib axis.
                    """
                    if ax is None:
                        fig, ax = plt.subplots()
                    
                    ax.plot(self._values, **kwargs)
                    ax.set_xlabel('Time')
                    ax.set_ylabel(self._name)
                    
                    return ax
                
                def __getitem__(self, key: int) -> Any:
                    """Get value at time step.
                    
                    Args:
                        key: Time step.
                    
                    Returns:
                        Value at time step.
                    """
                    return self._values[key]
                
                def __len__(self) -> int:
                    """Get number of time steps.
                    
                    Returns:
                        Number of time steps.
                    """
                    return len(self._values)
    
    def __init__(self, data: Dict[str, Any]):
        """Initialize results.
        
        Args:
            data: Simulation data.
        """
        self._data = data
        
        # Convert JAX arrays to numpy
        self._data = convert_to_numpy(self._data)
        
        # Create variable container
        self.variables = self.VariableContainer(self._data)
    
    def plot(self, variables=None, ax=None, **kwargs):
        """Plot results.
        
        Args:
            variables: List of variables to plot.
            ax: Matplotlib axis.
            **kwargs: Additional keyword arguments for plotting.
        
        Returns:
            Matplotlib axis.
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        if variables is None:
            # Plot all variables
            for key, values in self._data.items():
                if isinstance(values, list) and all(isinstance(v, (int, float)) for v in values):
                    ax.plot(values, label=key, **kwargs)
        else:
            # Plot specified variables
            for var in variables:
                if var in self._data:
                    ax.plot(self._data[var], label=var, **kwargs)
        
        ax.legend()
        ax.set_xlabel('Time')
        
        return ax


class Model:
    """Base class for agent-based models in JaxABM.
    
    This class provides an AgentPy-like interface for creating and running
    agent-based models.
    
    Example:
        ```python
        class MyModel(Model):
            def setup(self):
                self.agents = AgentList(self, 10, MyAgent)
                
            def step(self):
                self.agents.step()
                
        model = MyModel(parameters)
        results = model.run()
        ```
    """
    
    def __init__(self, parameters: Dict[str, Any] = None, seed: int = None):
        """Initialize model.
        
        Args:
            parameters: Model parameters.
            seed: Random seed.
        """
        self.p = parameters or {}
        
        # Set seed from parameters or argument
        self.seed = seed if seed is not None else self.p.get('seed', 0)
        
        # Set steps from parameters
        self.steps = self.p.get('steps', 100)
        
        # Initialize environment
        self.env = Environment(self)
        
        # Initialize data recording
        self._recorded_data = {}
        
        # Agent lists
        self._agent_lists = {}
        
        # JAX model (will be created in run)
        self._model = None
        
        # JAX model state update and metrics functions
        self._update_state_fn = None
        self._metrics_fn = None
    
    def setup(self) -> None:
        """Set up model.
        
        Override this method to set up agents and environment.
        """
        pass
    
    def step(self) -> None:
        """Execute a single time step.
        
        Override this method to define model behavior.
        """
        # Default behavior: step all agent lists
        for agent_list in self._agent_lists.values():
            agent_list.step()
    
    def update_state(self, env_state: Dict[str, Any], agent_states: Dict[str, Dict[str, Any]],
                    model_params: Dict[str, Any], key: jax.Array) -> Dict[str, Any]:
        """Update model state.
        
        This is the function that will be called by the JaxModel to update the model state.
        If you override this, it will be used when constructing the JaxModel.
        
        Args:
            env_state: Current environment state.
            agent_states: Current agent states by collection.
            model_params: Model parameters.
            key: JAX random key.
            
        Returns:
            Updated environment state.
        """
        return env_state
    
    def compute_metrics(self, env_state: Dict[str, Any], agent_states: Dict[str, Dict[str, Any]],
                       model_params: Dict[str, Any]) -> Dict[str, Any]:
        """Compute model metrics.
        
        This is the function that will be called by the JaxModel to compute metrics.
        If you override this, it will be used when constructing the JaxModel.
        
        Args:
            env_state: Current environment state.
            agent_states: Current agent states by collection.
            model_params: Model parameters.
            
        Returns:
            Dictionary of metrics.
        """
        return {}
    
    def add_agents(self, n: int, agent_class: Type[Agent], name: str = None, **kwargs) -> AgentList:
        """Add agents to the model.
        
        Args:
            n: Number of agents to add.
            agent_class: Agent class to use.
            name: Name for this agent collection.
            **kwargs: Parameters to pass to the agents.
            
        Returns:
            AgentList of created agents.
        """
        # Create agent list
        agent_list = AgentList(self, n, agent_class, **kwargs)
        
        # Generate name if not provided
        if name is None:
            name = agent_class.__name__.lower() + 's'
        
        # Store agent list
        self._agent_lists[name] = agent_list
        agent_list.name = name
        
        return agent_list
    
    def record(self, name: str, value: Any) -> None:
        """Record data for later analysis.
        
        Args:
            name: Name for the recorded data.
            value: Value to record.
        """
        self._recorded_data.setdefault(name, []).append(value)
    
    def run(self, steps: int = None) -> Results:
        """Run the model for the specified number of steps.
        
        Args:
            steps: Number of steps to run.
            
        Returns:
            Results object containing simulation results.
        """
        # Set steps from argument or parameters
        if steps is not None:
            self.steps = steps
        
        # Create model config
        config = ModelConfig(
            steps=self.steps,
            collect_interval=1,
            seed=self.seed
        )
        
        # Initialize the model
        self.setup()
        
        # Check if we have update_state and metrics functions
        if self._update_state_fn is None:
            # Create update_state_fn from instance method
            self._update_state_fn = self.update_state
        
        if self._metrics_fn is None:
            # Create metrics_fn from instance method
            self._metrics_fn = self.compute_metrics
        
        # Create JAX model
        self._model = JaxModel(
            params=self.p,
            config=config,
            update_state_fn=self._update_state_fn,
            metrics_fn=self._metrics_fn
        )
        
        # Add agent collections to the model
        for name, agent_list in self._agent_lists.items():
            self._model.add_agent_collection(name, agent_list.collection)
        
        # Run the JAX model
        start_time = time.time()
        results_dict = self._model.run()
        end_time = time.time()
        
        # Prepare results
        results_dict.update(self._recorded_data)
        
        # Format simulation time
        simulation_time = end_time - start_time
        
        # Add agent states to results
        if self._model.state and 'agents' in self._model.state:
            for name, states in self._model.state['agents'].items():
                for state_name, values in states.items():
                    results_dict[f'agents.{name}.{state_name}'] = values
        
        # Create and return Results object
        results = Results(results_dict)
        
        # Print execution time
        print(f"Simulation executed in {format_time(simulation_time)}")
        
        return results

# Make classes available at the module level
__all__ = ['Agent', 'AgentList', 'Environment', 'Model', 'Results']