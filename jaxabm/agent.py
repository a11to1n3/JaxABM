"""
Core abstractions for agents in the JaxABM framework.

This module defines the key abstractions for working with agents in
JAX-accelerated agent-based models, including agent types and collections.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, List, Tuple, Protocol, Callable, Optional, Union

# Import ModelConfig from core
from .core import ModelConfig

# Type alias for agent states
AgentState = Dict[str, Any]


class AgentType(Protocol):
    """Protocol for agent types.
    
    This protocol defines the interface that agent types must implement.
    Agent types are responsible for initializing and updating agent states.
    """
    
    def init_state(self, model_config: Any, key: Any) -> AgentState:
        """Initialize agent state.
        
        Args:
            model_config: Model configuration
            key: Random key for stochastic initialization
            
        Returns:
            Initial agent state
        """
        ...
        
    def update(self, state: AgentState, model_state: Dict[str, Any], 
              model_config: Any, key: Any) -> AgentState:
        """Update agent state.
        
        Args:
            state: Current agent state
            model_state: Current model state
            model_config: Model configuration
            key: Random key for stochastic updates
            
        Returns:
            Updated agent state
        """
        ...


class AgentCollection:
    """Collection of agents of the same type.
    
    This class manages a collection of agents of the same type, providing
    methods for initialization, updating, and accessing agent states.
    
    Attributes:
        agent_type: The type of agent in the collection
        num_agents: Number of agents in the collection
        model_config: Model configuration associated with this collection (set during Model.initialize)
        _states: Dictionary of agent state variables, with each variable
            having shape (num_agents, ...)
        _key: The JAX PRNGKey used to initialize this collection
    """
    
    def __init__(
        self, 
        agent_type: AgentType, 
        num_agents: int,
    ):
        """Initialize agent collection placeholder.

        The actual state initialization happens during the `init` method,
        which is typically called by `Model.initialize()`.
        
        Args:
            agent_type: The type of agent in this collection (must adhere to AgentType protocol)
            num_agents: The number of agents to create in this collection.
        """
        if not isinstance(num_agents, int) or num_agents <= 0:
            raise ValueError("num_agents must be a positive integer")
            
        self.agent_type = agent_type
        self.num_agents = num_agents
        self.model_config: Optional[ModelConfig] = None # Set later by Model.initialize
        self._key: Optional[jax.Array] = None # Set later by Model.initialize
        self._states: Optional[Dict[str, jax.Array]] = None
        
    def init(self, key: Any, model_config: ModelConfig) -> None:
        """Initialize agent states.
        
        This method initializes the states of all agents in the collection
        using the agent type's `init_state` method. It is typically called
        by `Model.initialize()`.
        
        Args:
            key: Random key for stochastic initialization.
            model_config: Model configuration settings passed from the Model.
        """
        if not isinstance(model_config, ModelConfig):
            raise TypeError("model_config must be a ModelConfig instance.")
        
        # Check for num_agents - should always be set in __init__ now
        if not isinstance(self.num_agents, int) or self.num_agents <= 0:
            raise ValueError("Number of agents must be a positive integer.")
             
        # Store the key and config for later use (e.g., in update if needed)
        self._key = key
        self.model_config = model_config
        
        # Create individual keys for each agent
        agent_keys = jax.random.split(key, self.num_agents)
        
        # Get the mandatory init_state method from the agent type
        init_method = getattr(self.agent_type, 'init_state', None)
        if not callable(init_method):
             raise AttributeError(f"Agent type {self.agent_type.__name__} must implement 'init_state'")
        
        # Initialize each agent's state using the protocol method
        # Vectorize the initialization function over agent keys
        # The vmap will return a dictionary where each key maps to an array of values
        init_vmap = jax.vmap(lambda k: init_method(model_config, k))
        initialized_states = init_vmap(agent_keys)
        
        # JAX's vmap returns a dictionary of arrays, where each array has shape (batch_size, ...),
        # so we can directly use it as our states
        self._states = initialized_states
        
    def update(
        self, 
        model_state: Dict[str, Any], 
        key: Any,
        model_config: ModelConfig 
    ) -> None: # Now returns None, state is updated internally
        """Update all agents in the collection using JAX vmap.
        
        This method updates the internal states (`self._states`) of all agents 
        using their agent type's `update` method, vectorized with `jax.vmap`.
        It assumes the agent type adheres to the `AgentType` protocol and its
        `update` method returns only the updated `AgentState`.
        
        Args:
            model_state: Current model state (environment + other agent states)
            key: Random key for stochastic updates
            model_config: Model configuration settings
        """
        if self._states is None:
            raise ValueError("Agent collection not initialized. Call init() first.")
        if self.model_config is None: # Config should be set during init
             raise RuntimeError("Model config not set for AgentCollection. Ensure Model.initialize() was called.")
            
        # Split random keys for each agent
        agent_keys = jax.random.split(key, self.num_agents)
        
        # Access current states
        current_states = self._states
        
        # Get the mandatory update method from the agent type
        update_method = getattr(self.agent_type, 'update', None)
        if not callable(update_method):
             raise AttributeError(f"Agent type {self.agent_type.__name__} must implement 'update'")

        # Define a function that takes an individual agent state and key
        # JAX's vmap will apply this to each agent's state
        def agent_update_fn(agent_states, agent_key):
            return update_method(agent_states, model_state, model_config, agent_key)
        
        # Vectorize the update function over all agents
        # For each agent, it gets its own state dict and random key
        batched_update = jax.vmap(agent_update_fn)
        
        # Update all agents at once - the result is a dictionary of arrays
        # where each array has shape (num_agents, ...)
        self._states = batched_update(current_states, agent_keys)
        
        # This method no longer returns anything; state mutation happens internally.
    
    def get_states(self) -> Dict[str, Any]:
        """Get agent states (alias for states property for backward compatibility).
        
        Returns:
            Dictionary of agent state variables
        """
        return self._states
    
    @property
    def states(self) -> Dict[str, jnp.ndarray]:
        """Get agent states.
        
        Returns:
            Dictionary of agent state variables
        """
        return self._states
    
    def aggregate(self, variable: str, fn: Callable = jnp.mean) -> Any:
        """Aggregate a state variable across agents.
        
        Args:
            variable: Name of the state variable to aggregate
            fn: Aggregation function (default: mean)
            
        Returns:
            Aggregated value
        """
        if variable not in self._states:
            raise ValueError(f"Variable {variable} not found in agent states")
        
        return fn(self._states[variable])
    
    def filter(self, condition: Callable[[Dict[str, Any]], jnp.ndarray]) -> 'AgentCollection':
        """Filter agents based on a condition.
        
        This method creates a new agent collection with agents that meet
        the specified condition.
        
        Args:
            condition: Function that takes agent state and returns boolean array
            
        Returns:
            New agent collection with filtered agents
        """
        # Create a new collection with the same agent type
        # Create mask from condition
        agent_states = {k: self._states[k] for k in self._states}
        mask = condition(agent_states)
        filtered_count = int(jnp.sum(mask))
        
        # Create a new collection with the filtered count
        filtered = AgentCollection(
            agent_type=self.agent_type,
            num_agents=filtered_count
        )
        
        # Set the model_config and key directly
        filtered.model_config = self.model_config
        filtered._key = self._key
        
        # Apply mask to each state variable
        filtered._states = {k: v[mask] for k, v in self._states.items()}
        
        return filtered 