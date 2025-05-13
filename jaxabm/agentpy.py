"""
JaxABM Core Agent-based Modeling Classes with AgentPy-like Interface

This module provides the main classes for the JaxABM framework with an
AgentPy-like interface. This makes it easier to create, run, and analyze
agent-based models while maintaining the performance benefits of JAX.

The main classes are:
- Agent: Base class for creating agents
- AgentList: Container for managing collections of agents
- Environment: Container for environment state and spatial structures
- Model: Base class for creating models
- Results: Container for simulation results and analysis tools
- Parameter: Class for parameter definition (for sensitivity analysis)
- Sample: Class for parameter samples (for batch runs)
- SensitivityAnalyzer: Wrapper for sensitivity analysis
- ModelCalibrator: Wrapper for model calibration
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import warnings
import matplotlib.pyplot as plt
import os
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, TypeVar, Type, Set

# Import the core JAX-based components that we'll build upon
from .agent import AgentType, AgentCollection
from .core import ModelConfig
from .model import Model as JaxModel
from .utils import convert_to_numpy, format_time, run_parallel_simulations

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
                return {
                    'x': 0,
                    'y': 0
                }
                
            def step(self, model):
                x = self.x + 0.1
                y = self.y + 0.1
                return {
                    'x': x,
                    'y': y
                }
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
    
    def step(self, model_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Update agent state.
        
        Override this method to define agent behavior.
        
        Args:
            model_state: The current model state.
            
        Returns:
            A dictionary containing the updated agent state.
        """
        return self._state
    
    def __getattr__(self, name: str) -> Any:
        """Get agent attribute from state.
        
        This allows accessing state variables as attributes, e.g., agent.x
        instead of agent._state['x'].
        
        Args:
            name: Attribute name.
            
        Returns:
            Attribute value if it exists in _state.
        
        Raises:
            AttributeError: If attribute not found in _state.
        """
        if self._state and name in self._state:
            return self._state[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Set agent attribute.
        
        Special attributes (id, model, p, _state) are set normally.
        Other attributes are set in the _state dictionary.
        
        Args:
            name: Attribute name.
            value: Attribute value.
        """
        if name in ('id', 'model', 'p', '_state'):
            super().__setattr__(name, value)
        else:
            if not hasattr(self, '_state'):
                super().__setattr__('_state', {})
            self._state[name] = value


class AgentWrapper(AgentType):
    """Adapter that wraps Agent class to implement AgentType protocol."""
    
    def __init__(self, agent_class: Type[Agent], params: Optional[Dict[str, Any]] = None):
        """Initialize agent wrapper.
        
        Args:
            agent_class: The Agent class to wrap.
            params: Parameters to pass to the agent.
        """
        self.agent_class = agent_class
        self.params = params or {}
        
        # Create a template instance to access methods
        self.agent_instance = agent_class()
        if params:
            self.agent_instance.p = params
    
    def init_state(self, model_config: ModelConfig, key: jax.Array) -> Dict[str, Any]:
        """Initialize agent state.
        
        Implements AgentType protocol by calling the Agent.setup method.
        
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
        
        Implements AgentType protocol by calling the Agent.step method.
        
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
        # In Model.setup():
        self.agents = AgentList(self, 10, MyAgent)
        
        # Access agent attributes:
        x_positions = self.agents.x  # Returns array of x values
        
        # Filter agents:
        active_agents = self.agents.select(lambda agents: agents.active)
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
        
        # Create AgentWrapper and AgentCollection
        self.agent_type = AgentWrapper(agent_class, kwargs)
        self.collection = AgentCollection(
            agent_type=self.agent_type,
            num_agents=n
        )
        
        # Store name for use with model
        self.name = None
    
    @property
    def states(self) -> Dict[str, Any]:
        """Get all agent states.
        
        Returns:
            Dictionary of agent state variables.
        """
        # Try to get states from model
        if hasattr(self.model, '_jax_model') and self.model._jax_model.state:
            agent_states = self.model._jax_model.state.get('agents', {})
            if self.name and self.name in agent_states:
                return agent_states[self.name]
        # Fallback to collection
        if hasattr(self.collection, 'states'):
            return self.collection.states
        return {}
    
    def __getattr__(self, name: str) -> Any:
        """Get agent attribute for all agents.
        
        This allows getting arrays of attribute values, e.g., agents.x.
        
        Args:
            name: Attribute name.
            
        Returns:
            Array of attribute values.
            
        Raises:
            AttributeError: If attribute not found.
        """
        states = self.states
        if name in states:
            return states[name]
        raise AttributeError(f"'AgentList' object has no attribute '{name}'")
    
    def __len__(self) -> int:
        """Get number of agents.
        
        Returns:
            Number of agents.
        """
        return self.n
    
    def select(self, condition: Callable[[Any], Any]) -> 'AgentList':
        """Select agents that satisfy a condition.
        
        Args:
            condition: Function that takes agent attributes and returns 
                       a boolean array or mask.
            
        Returns:
            New AgentList with selected agents.
        """
        # Create a mask from the condition
        states = self.states
        mask = condition(states)
        
        # Count selected agents
        filtered_count = int(jnp.sum(mask))
        
        # Create new AgentList with same agent type
        filtered = AgentList(
            model=self.model,
            n=filtered_count,
            agent_class=self.agent_class,
            **self.params
        )
        
        # Filter the collection (if possible)
        if hasattr(self.collection, 'filter'):
            filtered.collection = self.collection.filter(lambda s: mask)
        
        return filtered
    
    def __iter__(self):
        """Allow iteration over agents.
        
        This is a convenience method that creates Agent instances on-the-fly
        with the appropriate state. For large agent counts, this is inefficient
        and should be avoided in favor of vectorized operations.
        
        Yields:
            Agent instances with appropriate state.
        """
        # Get all states
        states = self.states
        if not states:
            return
        
        # Get an arbitrary state variable to determine agent count
        any_state = next(iter(states.values()))
        num_agents = len(any_state)
        
        # Create agent instances with appropriate state
        for i in range(num_agents):
            agent = self.agent_class()
            agent.model = self.model
            agent.id = i
            agent.p = self.params
            agent._state = {k: states[k][i] for k in states}
            yield agent


class Environment:
    """Environment for agent interactions.
    
    This class provides a container for environment state and methods for
    creating and managing spatial structures like grids and networks.
    
    Example:
        ```python
        # In Model.setup():
        self.env.add_state('temperature', 25.0)
        
        # Access environment state:
        temp = self.env.temperature
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
        if hasattr(self.model, '_jax_model') and self.model._jax_model:
            self.model._jax_model.add_env_state(name, value)
    
    def __getattr__(self, name: str) -> Any:
        """Get environment state variable.
        
        Args:
            name: State variable name.
            
        Returns:
            State variable value.
            
        Raises:
            AttributeError: If state variable not found.
        """
        if name in self.state:
            return self.state[name]
        
        # Check model state if environment state isn't available
        if hasattr(self.model, '_jax_model') and self.model._jax_model and self.model._jax_model.state:
            env_state = self.model._jax_model.state.get('env', {})
            if name in env_state:
                return env_state[name]
        
        raise AttributeError(f"'Environment' object has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Set environment attribute.
        
        Special attributes (model, state) are set normally.
        Other attributes are set in the state dictionary and
        the underlying JAX model if it exists.
        
        Args:
            name: Attribute name.
            value: Attribute value.
        """
        if name in ('model', 'state'):
            super().__setattr__(name, value)
        else:
            if not hasattr(self, 'state'):
                super().__setattr__('state', {})
            self.state[name] = value
            
            # Update JAX model if it exists
            if hasattr(self, 'model') and hasattr(self.model, '_jax_model') and self.model._jax_model:
                self.model._jax_model.add_env_state(name, value)


class Grid:
    """Grid environment for spatial agent-based models.
    
    This class provides a 2D grid for agent interactions.
    
    Example:
        ```python
        # In Model.setup():
        self.grid = Grid(self, (10, 10))
        
        # Position agents on grid:
        self.grid.position_agents(self.agents)
        ```
    """
    
    def __init__(self, model: 'Model', shape: Tuple[int, int], periodic: bool = False):
        """Initialize grid.
        
        Args:
            model: The model the grid belongs to.
            shape: Grid shape (width, height).
            periodic: Whether the grid has periodic boundaries.
        """
        self.model = model
        self.shape = shape
        self.periodic = periodic
        
        # Add grid to model environment
        self.model.env.add_state('grid_shape', shape)
        self.model.env.add_state('grid_periodic', periodic)
    
    def position_agents(self, agents: AgentList, positions: Optional[jnp.ndarray] = None) -> None:
        """Position agents on the grid.
        
        Args:
            agents: The agents to position.
            positions: Optional array of positions (x, y) for each agent.
                      If not provided, agents are positioned randomly.
        """
        n = len(agents)
        width, height = self.shape
        
        # Generate random positions if not provided
        if positions is None:
            key = jax.random.PRNGKey(self.model.p.get('seed', 0))
            x = jax.random.randint(key, (n,), 0, width)
            key, subkey = jax.random.split(key)
            y = jax.random.randint(subkey, (n,), 0, height)
            positions = jnp.column_stack((x, y))
        
        # Update agent states
        if hasattr(agents.collection, 'states') and agents.collection.states is not None:
            # Check if position already exists in states
            states = agents.collection.states
            if 'position' in states:
                # We can't directly modify JAX arrays, so we create a new states dict
                new_states = {k: v for k, v in states.items()}
                new_states['position'] = positions
                agents.collection._states = new_states
            else:
                # Add position to states
                if hasattr(agents.collection, '_states'):
                    agents.collection._states['position'] = positions


class Network:
    """Network environment for agent interactions.
    
    This class provides a network structure for agent interactions.
    
    Example:
        ```python
        # In Model.setup():
        self.network = Network(self)
        
        # Add edges:
        self.network.add_edge(agent1, agent2)
        ```
    """
    
    def __init__(self, model: 'Model', directed: bool = False):
        """Initialize network.
        
        Args:
            model: The model the network belongs to.
            directed: Whether the network is directed.
        """
        self.model = model
        self.directed = directed
        
        # Add network to model environment
        self.model.env.add_state('network_directed', directed)
        self.model.env.add_state('network_edges', jnp.zeros((0, 2), dtype=jnp.int32))
    
    def add_edge(self, from_agent: Union[Agent, int], to_agent: Union[Agent, int]) -> None:
        """Add an edge to the network.
        
        Args:
            from_agent: The source agent or agent ID.
            to_agent: The target agent or agent ID.
        """
        # Convert agents to IDs if necessary
        from_id = from_agent.id if isinstance(from_agent, Agent) else from_agent
        to_id = to_agent.id if isinstance(to_agent, Agent) else to_agent
        
        # Get current edges
        current_edges = self.model.env.network_edges
        
        # Add new edge
        new_edge = jnp.array([[from_id, to_id]], dtype=jnp.int32)
        new_edges = jnp.concatenate([current_edges, new_edge], axis=0)
        
        # Update model environment
        self.model.env.add_state('network_edges', new_edges)
        
        # Add reverse edge for undirected networks
        if not self.directed and from_id != to_id:
            self.add_edge(to_id, from_id)
    
    def get_neighbors(self, agent: Union[Agent, int]) -> jnp.ndarray:
        """Get neighbors of an agent.
        
        Args:
            agent: The agent or agent ID.
            
        Returns:
            Array of neighbor agent IDs.
        """
        # Convert agent to ID if necessary
        agent_id = agent.id if isinstance(agent, Agent) else agent
        
        # Get edges
        edges = self.model.env.network_edges
        
        # Find neighbors
        if self.directed:
            # Only outgoing edges
            mask = edges[:, 0] == agent_id
            neighbors = edges[mask, 1]
        else:
            # Both incoming and outgoing edges
            mask1 = edges[:, 0] == agent_id
            mask2 = edges[:, 1] == agent_id
            neighbors1 = edges[mask1, 1]
            neighbors2 = edges[mask2, 0]
            neighbors = jnp.concatenate([neighbors1, neighbors2], axis=0)
            
            # Remove duplicates
            neighbors = jnp.unique(neighbors)
        
        return neighbors


class Results:
    """Container for simulation results.
    
    This class provides an AgentPy-like interface for accessing and
    visualizing simulation results.
    
    Example:
        ```python
        results = model.run()
        
        # Plot results:
        results.plot()
        
        # Access specific variables:
        results.variables.MyAgent.x.plot()
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
                self._variables = set()
                for key in data:
                    if key.startswith(f'agents.{agent_type}.'):
                        var_name = key.split('.')[-1]
                        self._variables.add(var_name)
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
                    
                    # Handle different data types
                    if self._values and hasattr(self._values[0], 'shape') and len(self._values[0].shape) > 0:
                        # Matrix data - plot mean
                        data = np.mean(np.array(self._values), axis=1)
                        ax.plot(data, **kwargs)
                        ax.set_ylabel(f'Mean {self._name}')
                    else:
                        # Scalar data
                        ax.plot(self._values, **kwargs)
                        ax.set_ylabel(self._name)
                    
                    ax.set_xlabel('Time')
                    
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
    
    def plot(self, variables: Optional[List[str]] = None, ax=None, **kwargs):
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
            # Plot all scalar variables
            for key, values in self._data.items():
                if isinstance(values, list) and all(isinstance(v, (int, float, np.number)) for v in values):
                    ax.plot(values, label=key, **kwargs)
        else:
            # Plot specified variables
            for var in variables:
                if var in self._data:
                    ax.plot(self._data[var], label=var, **kwargs)
        
        ax.legend()
        ax.set_xlabel('Time')
        
        return ax
    
    def save(self, filename: str) -> None:
        """Save results to file.
        
        Args:
            filename: Filename to save results to.
        """
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self._data, f)
    
    @classmethod
    def load(cls, filename: str) -> 'Results':
        """Load results from file.
        
        Args:
            filename: Filename to load results from.
            
        Returns:
            Results object.
        """
        import pickle
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return cls(data)


class Model:
    """Base class for agent-based models in JaxABM.
    
    This class provides an AgentPy-like interface for creating and running
    agent-based models.
    
    Example:
        ```python
        class MyModel(Model):
            def setup(self):
                self.agents = self.add_agents(10, MyAgent)
                self.env.temperature = 25.0
                
            def step(self):
                # Agents are stepped automatically by default
                # Add additional model logic here
                self.env.temperature += 0.1
                
                # Record data
                self.record('temperature', self.env.temperature)
                
        model = MyModel(parameters)
        results = model.run()
        ```
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None, seed: Optional[int] = None):
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
        
        # JAX model instance (initialized during run)
        self._jax_model = None
        
        # Initialize state
        self._dynamic_state_update = False
    
    def setup(self) -> None:
        """Set up model.
        
        Override this method to set up agents and environment.
        """
        pass
    
    def step(self) -> None:
        """Execute a single time step.
        
        Override this method to define model behavior.
        By default, it steps all agent lists.
        """
        # Default behavior: Let JAX model handle agent stepping
        pass
    
    def end(self) -> None:
        """Execute code at the end of a simulation.
        
        Override this method to define behavior at the end of a simulation.
        """
        pass
    
    def update_state(self, env_state: Dict[str, Any], agent_states: Dict[str, Dict[str, Any]],
                    model_params: Dict[str, Any], key: jax.Array) -> Dict[str, Any]:
        """Update model environment state.
        
        This method is called by the JAX model to update the environment state
        based on agent states. By default, it returns the environment state unchanged.
        Override this method to define custom state update logic.
        
        Args:
            env_state: Current environment state.
            agent_states: Current agent states by collection.
            model_params: Model parameters.
            key: JAX random key.
            
        Returns:
            Updated environment state.
        """
        # Store the current state for access in step
        self._current_env_state = env_state.copy()
        self._current_agent_states = agent_states
        
        # Call user-defined step function
        self.step()
        
        # Create new environment state from current + local env state
        new_env_state = {**env_state}
        for name, value in self.env.state.items():
            new_env_state[name] = value
            
        return new_env_state
    
    def compute_metrics(self, env_state: Dict[str, Any], agent_states: Dict[str, Dict[str, Any]],
                       model_params: Dict[str, Any]) -> Dict[str, Any]:
        """Compute model metrics.
        
        This method is called by the JAX model to compute metrics from model state.
        By default, it returns an empty dictionary. Override this method to define
        custom metrics.
        
        Args:
            env_state: Current environment state.
            agent_states: Current agent states by collection.
            model_params: Model parameters.
            
        Returns:
            Dictionary of metrics.
        """
        return {}
    
    def add_agents(self, n: int, agent_class: Type[Agent], name: Optional[str] = None, **kwargs) -> AgentList:
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
    
    def run(self, steps: Optional[int] = None) -> Results:
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
        
        # Check if step method has been overridden
        self._dynamic_state_update = self.__class__.step != Model.step
        
        # Create JAX model
        self._jax_model = JaxModel(
            params=self.p,
            config=config,
            update_state_fn=self.update_state,
            metrics_fn=self.compute_metrics
        )
        
        # Add agent collections to the model
        for name, agent_list in self._agent_lists.items():
            self._jax_model.add_agent_collection(name, agent_list.collection)
        
        # Add environment state to the model
        for name, value in self.env.state.items():
            self._jax_model.add_env_state(name, value)
        
        # Run the JAX model
        start_time = time.time()
        results_dict = self._jax_model.run()
        end_time = time.time()
        
        # Call end method
        self.end()
        
        # Prepare results
        results_dict.update(self._recorded_data)
        
        # Format simulation time
        simulation_time = end_time - start_time
        
        # Add agent states to results
        if self._jax_model.state and 'agents' in self._jax_model.state:
            for name, states in self._jax_model.state['agents'].items():
                for state_name, values in states.items():
                    results_dict[f'agents.{name}.{state_name}'] = values
        
        # Create and return Results object
        results = Results(results_dict)
        
        # Print execution time
        print(f"Simulation executed in {format_time(simulation_time)}")
        
        return results
    
    def batch_run(self, parameter_ranges: Dict[str, List[Any]], repetitions: int = 1) -> Dict[str, Results]:
        """Run model with multiple parameter combinations.
        
        Args:
            parameter_ranges: Dictionary mapping parameter names to lists of values.
            repetitions: Number of repetitions for each parameter combination.
            
        Returns:
            Dictionary mapping parameter combinations to Results objects.
        """
        # Generate all parameter combinations
        import itertools
        param_names = list(parameter_ranges.keys())
        param_values = list(itertools.product(*parameter_ranges.values()))
        
        results = {}
        
        # Run each parameter combination
        for values in param_values:
            # Create parameter dictionary
            params = {**self.p}  # Start with current parameters
            for name, value in zip(param_names, values):
                params[name] = value
            
            # Run repetitions
            rep_results = []
            for rep in range(repetitions):
                # Create new model with these parameters
                model = self.__class__(params, seed=self.seed + rep)
                # Run model
                rep_results.append(model.run())
            
            # Store results
            params_tuple = tuple(values)
            results[params_tuple] = rep_results
        
        return results


# Make classes available at the module level
__all__ = ['Agent', 'AgentList', 'Environment', 'Grid', 'Network', 'Model', 'Results',
           'Parameter', 'Sample', 'SensitivityAnalyzer', 'ModelCalibrator']


class Parameter:
    """Parameter for sensitivity analysis and model calibration.
    
    This class defines a parameter with a range of possible values,
    which can be used for sensitivity analysis or parameter calibration.
    
    Example:
        ```python
        # Create a parameter for sensitivity analysis
        p1 = Parameter('growth_rate', bounds=(0.01, 0.1))
        
        # Create a parameter with a distribution
        p2 = Parameter('initial_population', bounds=(10, 1000), 
                      distribution='uniform')
        ```
    """
    
    def __init__(self, name: str, bounds: Tuple[float, float], 
                 distribution: str = 'uniform'):
        """Initialize parameter.
        
        Args:
            name: Parameter name.
            bounds: Parameter bounds (min, max).
            distribution: Distribution for sampling ('uniform', 'normal', etc.).
        """
        self.name = name
        self.bounds = bounds
        self.distribution = distribution
    
    def sample(self, n: int = 1) -> np.ndarray:
        """Sample parameter values.
        
        Args:
            n: Number of samples.
            
        Returns:
            Array of sampled values.
        """
        # Use numpy for simple random sampling
        if self.distribution == 'uniform':
            return np.random.uniform(self.bounds[0], self.bounds[1], size=n)
        elif self.distribution == 'normal':
            mean = (self.bounds[0] + self.bounds[1]) / 2
            std = (self.bounds[1] - self.bounds[0]) / 4  # Approximate
            return np.random.normal(mean, std, size=n)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")


class Sample:
    """Container for parameter samples.
    
    This class stores parameter samples for batch runs or
    sensitivity analysis.
    
    Example:
        ```python
        # Create parameters
        p1 = Parameter('growth_rate', (0.01, 0.1))
        p2 = Parameter('initial_population', (10, 1000))
        
        # Create sample
        sample = Sample([p1, p2], n_samples=10)
        
        # Run model with sample
        results = analyzer.run(sample)
        ```
    """
    
    def __init__(self, parameters: List[Parameter], n_samples: int = 10):
        """Initialize sample.
        
        Args:
            parameters: List of parameters.
            n_samples: Number of samples per parameter.
        """
        self.parameters = parameters
        self.n_samples = n_samples
        
        # Sample parameter values
        self._samples = {}
        # Use numpy for random sampling
        
        for param in parameters:
            self._samples[param.name] = param.sample(n_samples)
    
    def __getitem__(self, index: int) -> Dict[str, float]:
        """Get parameter set at index.
        
        Args:
            index: Sample index.
            
        Returns:
            Dictionary of parameter values.
        """
        params = {}
        for param in self.parameters:
            params[param.name] = self._samples[param.name][index]
        return params
    
    def __len__(self) -> int:
        """Get number of samples.
        
        Returns:
            Number of samples.
        """
        return self.n_samples


class SensitivityAnalyzer:
    """Wrapper for sensitivity analysis with AgentPy-like interface.
    
    This class provides a more user-friendly interface for sensitivity
    analysis with JaxABM.
    
    Example:
        ```python
        # Create parameters
        p1 = Parameter('growth_rate', (0.01, 0.1))
        p2 = Parameter('initial_population', (10, 1000))
        
        # Create analyzer
        analyzer = SensitivityAnalyzer(
            MyModel,
            parameters=[p1, p2],
            n_samples=10,
            metrics=['population', 'resources']
        )
        
        # Run analysis
        results = analyzer.run()
        
        # Calculate sensitivity
        sensitivity = analyzer.calculate_sensitivity()
        ```
    """
    
    def __init__(self, model_class: Type[Model], parameters: List[Parameter],
                n_samples: int = 10, metrics: List[str] = None):
        """Initialize sensitivity analyzer.
        
        Args:
            model_class: Model class to analyze.
            parameters: List of parameters to vary.
            n_samples: Number of samples per parameter.
            metrics: List of metrics to analyze.
        """
        self.model_class = model_class
        self.parameters = parameters
        self.n_samples = n_samples
        self.metrics = metrics or []
        
        # Create sample
        self.sample = Sample(parameters, n_samples)
        
        # Import actual sensitivity analysis
        from .analysis import SensitivityAnalysis
        
        # Create model factory function
        def model_factory(params=None, config=None):
            """Create model instance with parameters."""
            model = self.model_class(params)
            return model
        
        # Create parameter ranges for sensitivity analysis
        param_ranges = {}
        for param in parameters:
            param_ranges[param.name] = param.bounds
        
        # Create sensitivity analysis
        self.analysis = SensitivityAnalysis(
            model_factory=model_factory,
            param_ranges=param_ranges,
            metrics_of_interest=metrics,
            num_samples=n_samples
        )
    
    def run(self) -> Dict[str, Any]:
        """Run sensitivity analysis.
        
        Returns:
            Dictionary of results.
        """
        results_obj = self.analysis.run()
        
        # Convert Results object to dictionary if needed
        if hasattr(results_obj, '_data'):
            return results_obj._data
        return results_obj
    
    def calculate_sensitivity(self, method: str = 'sobol') -> Dict[str, Any]:
        """Calculate sensitivity indices.
        
        Args:
            method: Method for calculating sensitivity indices.
                   'sobol' or 'morris'.
            
        Returns:
            Dictionary of sensitivity indices.
        """
        if method == 'sobol':
            return self.analysis.sobol_indices()
        elif method == 'morris':
            return self.analysis.morris_indices()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def plot(self, metric: Optional[str] = None, ax=None, **kwargs):
        """Plot sensitivity analysis results.
        
        Args:
            metric: Metric to plot. If None, plot all metrics.
            ax: Matplotlib axis.
            **kwargs: Additional keyword arguments for plotting.
            
        Returns:
            Matplotlib axis.
        """
        # For now, we'll just call the underlying analysis plot method
        return self.analysis.plot(metric, ax, **kwargs)


class ModelCalibrator:
    """Wrapper for model calibration with AgentPy-like interface.
    
    This class provides a more user-friendly interface for model
    calibration with JaxABM.
    
    Example:
        ```python
        # Create parameters
        p1 = Parameter('growth_rate', (0.01, 0.1))
        p2 = Parameter('initial_population', (10, 1000))
        
        # Create calibrator
        calibrator = ModelCalibrator(
            MyModel,
            parameters=[p1, p2],
            target_metrics={'population': 500, 'resources': 1000},
            metrics_weights={'population': 1.0, 'resources': 0.5}
        )
        
        # Run calibration
        optimal_params = calibrator.run()
        ```
    """
    
    def __init__(self, model_class: Type[Model], parameters: List[Parameter],
                target_metrics: Dict[str, float], metrics_weights: Dict[str, float] = None,
                learning_rate: float = 0.01, max_iterations: int = 20,
                method: str = 'gradient'):
        """Initialize model calibrator.
        
        Args:
            model_class: Model class to calibrate.
            parameters: List of parameters to optimize.
            target_metrics: Dictionary of target metrics.
            metrics_weights: Dictionary of metric weights for loss function.
            learning_rate: Learning rate for optimization.
            max_iterations: Maximum number of iterations.
            method: Optimization method ('gradient' or 'rl').
        """
        self.model_class = model_class
        self.parameters = parameters
        self.target_metrics = target_metrics
        self.metrics_weights = metrics_weights or {m: 1.0 for m in target_metrics}
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.method = method
        
        # Import actual model calibrator
        from .analysis import ModelCalibrator as JaxModelCalibrator
        
        # Create model factory function
        def model_factory(params=None, config=None):
            """Create model instance with parameters."""
            model = self.model_class(params)
            return model
        
        # Create initial parameters
        initial_params = {}
        for param in parameters:
            # Start with middle of range
            initial_params[param.name] = (param.bounds[0] + param.bounds[1]) / 2
        
        # Create calibrator
        self.calibrator = JaxModelCalibrator(
            model_factory=model_factory,
            initial_params=initial_params,
            target_metrics=target_metrics,
            metrics_weights=metrics_weights,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            method=method
        )
    
    def run(self) -> Dict[str, float]:
        """Run calibration.
        
        Returns:
            Dictionary of optimized parameters.
        """
        return self.calibrator.calibrate()
    
    def plot_progress(self, ax=None, **kwargs):
        """Plot calibration progress.
        
        Args:
            ax: Matplotlib axis.
            **kwargs: Additional keyword arguments for plotting.
            
        Returns:
            Matplotlib axis.
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        # Plot loss over iterations
        if hasattr(self.calibrator, 'loss_history') and self.calibrator.loss_history:
            ax.plot(self.calibrator.loss_history, **kwargs)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
            ax.set_title('Calibration Progress')
            ax.grid(True)
        
        return ax