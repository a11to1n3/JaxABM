"""
Core model framework for the JaxABM agent-based modeling library.

This module provides the Model class that serves as the central component
for building JAX-accelerated agent-based models.
"""

import jax
import jax.numpy as jnp
import time
from typing import Dict, Any, List, Callable, Optional, Tuple

# Import AgentCollection from .agent and ModelConfig from .core
from .agent import AgentCollection 
from .core import ModelConfig


class Model:
    """Core model class for the JaxABM framework.
    
    The Model class coordinates agent collections, environmental state,
    and model dynamics in a JAX-accelerated agent-based model.
    """
    
    def __init__(
        self, 
        params: Optional[Dict[str, Any]] = None,
        config: Optional[ModelConfig] = None,
        update_state_fn: Optional[Callable[[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, Any], jax.Array], Dict[str, Any]]] = None,
        metrics_fn: Optional[Callable[[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        """Initialize a new model.

        Agent collections and environment state variables should be added
        using `add_agent_collection` and `add_env_state` after initialization.
        
        Args:
            params: Model parameters (static during simulation)
            config: Model configuration settings (seed, steps, etc.)
            update_state_fn: Function to update environment state based on agent states.
                             Signature: `(env_state, agent_states, params, rng) -> new_env_state`
            metrics_fn: Function to compute metrics from model state.
                          Signature: `(env_state, agent_states, params) -> metrics_dict`
        """
        self.config = config or ModelConfig()
        self._rng = jax.random.PRNGKey(self.config.seed)
        
        # Initialize collections, state, and parameters
        self._agent_collections: Dict[str, AgentCollection] = {}
        self._env_state: Dict[str, Any] = {}
        self._state: Optional[Dict[str, Any]] = None
        self._params = params or {}
        self._update_state_fn = update_state_fn
        self._metrics_fn = metrics_fn
        
        self._time_step = 0
        self._history = []
        self._is_initialized = False
    
    def add_agent_collection(
        self, 
        name: str, 
        agent_collection: AgentCollection
    ) -> None:
        """Add an agent collection to the model.
        
        Args:
            name: Name for the agent collection
            agent_collection: The agent collection to add
        """
        if self._is_initialized:
            raise RuntimeError("Cannot add agent collections after model is initialized")
        
        self._agent_collections[name] = agent_collection
    
    def add_env_state(self, name: str, value: Any) -> None:
        """Add a state variable to the model environment.
        
        Args:
            name: Name of the state variable.
            value: Value of the state variable.
        """
        # Update the state if model is initialized
        if self._is_initialized:
            # Just update the state
            if self._state is not None:
                if 'env' not in self._state:
                    self._state['env'] = {}
                self._state['env'][name] = value
            
            # Also update the env_state dictionary
            if self._env_state is None:
                self._env_state = {}
            self._env_state[name] = value
        else:
            # Initial state setup
            if self._env_state is None:
                self._env_state = {}
            self._env_state[name] = value
    
    def model_state(self) -> Dict[str, Any]:
        """Get the current model state.
        
        Returns:
            Dictionary containing current environmental state and time step
        """
        state = {
            "time_step": self._time_step,
            "env": self._env_state
        }
        
        # Add agent collection states
        for name, collection in self._agent_collections.items():
            state[f"agents_{name}"] = collection.states
        
        return state
    
    def initialize(self) -> None:
        """Initialize the model by initializing all agent collections.
        
        This method prepares the model for simulation by initializing all agent
        collections with random keys derived from the model's PRNG.
        It must be called before running `step` or `run`.
        """
        if not self._agent_collections:
            raise ValueError("No agent collections added to model")
        
        # Split random keys for each agent collection
        keys = jax.random.split(self._rng, len(self._agent_collections) + 1)
        self._rng = keys[0]
        
        # Initialize each agent collection
        for i, collection in enumerate(self._agent_collections.values()):
            # Ensure collection has access to model config if needed for init
            if not hasattr(collection, 'model_config') or collection.model_config is None:
                 collection.model_config = self.config
            collection.init(keys[i + 1], self.config)
        
        self._is_initialized = True
    
    def step(self) -> Dict[str, Any]:
        """Execute a single time step of the model.
        
        Returns:
            Dictionary containing metrics calculated for this step.
        """
        if not self._is_initialized:
            raise RuntimeError("Model must be initialized before stepping. Call initialize() first.")
        
        # Split a new random key for this step
        self._rng, step_key = jax.random.split(self._rng)
        
        # --- Agent Update Phase --- 
        # Get model state *before* agent updates (needed by agents)
        current_model_state = self.model_state()
        agent_outputs = {} # Store any outputs agents might produce (optional)

        for name, collection in self._agent_collections.items():
            step_key, coll_key = jax.random.split(step_key)
            # AgentCollection.update modifies internal state and returns Nothing
            # Pass model_state, key, and model_config
            collection.update(
                current_model_state, 
                coll_key,
                self.config 
            )
            agent_outputs[name] = None # No outputs from update
        
        # --- State Update Phase --- 
        # Collect the updated agent states *after* all agents have acted
        updated_agent_states = {
            name: collection.states 
            for name, collection in self._agent_collections.items()
        }

        # Update environment state using the provided function
        if self._update_state_fn:
            step_key, update_key = jax.random.split(step_key)
            # Pass current env state, new agent states, params, and key
            self._env_state = self._update_state_fn(
                self._env_state, 
                updated_agent_states, 
                self._params, 
                update_key
            )
            
        # --- Metrics Calculation Phase --- 
        metrics = {}
        if self._metrics_fn:
            # Pass the new env state and new agent states
            metrics = self._metrics_fn(
                self._env_state, 
                updated_agent_states, 
                self._params
            )
        
        # Increment time step
        self._time_step += 1
        
        # --- History Tracking --- 
        if self.config.track_history and self._time_step % self.config.collect_interval == 0:
            # Store metrics from this step. Avoid storing full state usually.
            self._history.append({
                "time_step": self._time_step, 
                # "state": self._env_state, # Typically too large
                # "agent_states": updated_agent_states, # Typically too large
                "metrics": metrics
            })
        
        # Return metrics calculated for this step
        return metrics
    
    def run(self, steps: Optional[int] = None) -> Dict[str, List[Any]]:
        """Run the model for a specified number of steps.
        
        Args:
            steps: Number of steps to run (overrides config.steps if provided)
            
        Returns:
            Dictionary of metrics with their values at each collected time step
        """
        if not self._is_initialized:
            self.initialize()
        
        # Use steps from parameters or config
        steps_to_run = steps if steps is not None else self.config.steps
        
        # Reset history if tracking
        if self.config.track_history:
            self._history = []
        
        start_time = time.time()
        
        # Run the simulation
        for _ in range(steps_to_run):
            self.step()
            
        elapsed = time.time() - start_time
        steps_per_second = steps_to_run / elapsed
        
        print(f"Ran {steps_to_run} steps in {elapsed:.2f}s ({steps_per_second:.1f} steps/sec)")
        
        # Process and return collected metrics
        if self.config.track_history and self._history:
            # Restructure history for easier access
            metrics = {}
            
            # Add step numbers to metrics
            metrics['step'] = [h["time_step"] for h in self._history]
            
            # Add all metrics
            for metric_name in self._history[0]["metrics"].keys():
                metrics[metric_name] = [h["metrics"][metric_name] for h in self._history]
                
            return metrics
        
        return {}
    
    @property
    def agent_collections(self) -> Dict[str, AgentCollection]:
        """Get the agent collections in the model.
        
        Returns:
            Dictionary of agent collections
        """
        return self._agent_collections
    
    @property
    def state(self) -> Dict[str, Any]:
        """Get the current environmental state.
        
        Returns:
            Environmental state dictionary
        """
        return self._env_state
    
    def jit_step(self) -> Callable:
        """JIT-compile the step function for improved performance.
        
        This method uses JAX's just-in-time compilation to optimize the step function,
        which can significantly improve performance for large models.
        
        Returns:
            JIT-compiled step function
        """
        # Define a pure function for JIT compilation
        # We need to extract the relevant state and create a pure function
        # that can be JIT-compiled
        
        def _step_fn(env_state, agent_states, params, key):
            # Similar logic to step, but as a pure function
            
            # --- Agent Update Phase ---
            # Create model state representation for agents
            model_state = {
                "time_step": self._time_step,
                "env": env_state
            }
            
            # For each agent collection
            updated_agent_states = {}
            for name, states in agent_states.items():
                # Get the collection
                collection = self._agent_collections[name]
                # Create a new random key
                key, subkey = jax.random.split(key)
                # Get the update method directly
                update_method = collection.agent_type.update
                # Apply it across all agents with vmap
                updated_states = jax.vmap(
                    lambda agent_state, agent_key: update_method(
                        agent_state, model_state, self.config, agent_key)
                )(states, jax.random.split(subkey, states['position'].shape[0]))
                updated_agent_states[name] = updated_states
            
            # --- State Update Phase ---
            # Update environment state using the provided function
            if self._update_state_fn:
                key, update_key = jax.random.split(key)
                env_state = self._update_state_fn(
                    env_state, updated_agent_states, params, update_key)
            
            # --- Metrics Calculation ---
            metrics = {}
            if self._metrics_fn:
                metrics = self._metrics_fn(
                    env_state, updated_agent_states, params)
                
            return env_state, updated_agent_states, metrics, key
        
        # JIT-compile the step function
        return jax.jit(_step_fn) 