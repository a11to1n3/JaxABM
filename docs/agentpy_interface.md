# JaxABM with AgentPy-like Interface

This guide shows how to use the new AgentPy-like interface in JaxABM, which provides a more user-friendly way to create agent-based models while maintaining the performance benefits of JAX.

## Basic Usage

The AgentPy-like interface can be used by importing the required classes from the `jaxabm` package:

```python
import jaxabm as jx
```

Alternatively, you can use the `ap` namespace to access the AgentPy-like components:

```python
import jaxabm as jx
from jaxabm import ap  # AgentPy-like namespace

# Now use ap.Agent, ap.Model, etc.
```

## Creating Agents

To create agents, inherit from the `Agent` class and override the `setup` and `step` methods:

```python
class MyAgent(jx.Agent):
    def setup(self):
        """Initialize agent state."""
        return {
            'x': 0,
            'y': 0
        }
    
    def step(self, model_state):
        """Update agent state."""
        # Get current state
        x = self._state['x']
        y = self._state['y']
        
        # Update state
        x += 1
        y += 1
        
        # Return updated state
        return {
            'x': x,
            'y': y
        }
```

## Creating Models

To create a model, inherit from the `Model` class and override the `setup` and `step` methods:

```python
class MyModel(jx.Model):
    def setup(self):
        """Set up model with agents and environment."""
        # Add agents
        self.agents = self.add_agents(
            n=self.p.get('n_agents', 10),
            agent_class=MyAgent
        )
        
        # Add environment variables
        self.env.add_state('time', 0)
    
    def step(self):
        """Execute a single time step."""
        # Additional model logic here
        # (agents are stepped automatically)
        
        # Update environment
        if hasattr(self._jax_model, 'state'):
            time = self._jax_model.state['env'].get('time', 0)
            self._jax_model.add_env_state('time', time + 1)
        
        # Record data
        self.record('time', time)
```

## Running Simulations

To run a simulation:

```python
# Define parameters
parameters = {
    'n_agents': 100,
    'steps': 100,
    'seed': 42
}

# Create and run model
model = MyModel(parameters)
results = model.run()

# Access and visualize results
results.plot()  # Plot all metrics
# Or access specific metrics
results.variables.agents.x.plot()  # Plot agent x values
```

## Key Differences from AgentPy

While the interface is similar to AgentPy, there are some differences to be aware of:

1. Agent state is represented as a dictionary returned from `setup` and `step` methods
2. JAX's constraints on mutation and randomness apply
3. The performance benefits come from JAX, which requires immutable data structures

## Examples

For complete examples, see:
- `examples/agentpy_interface_example.py` - A simple example with bouncing agents
- `examples/minimal_example_agentpy.py` - AgentPy-like version of the minimal example

## Comparison with Original JaxABM Interface

Original JaxABM Interface:
```python
class MyAgentType(AgentType):
    def init_state(self, model_config, key):
        return {'x': 0, 'y': 0}
    
    def update(self, state, model_state, model_config, key):
        return {'x': state['x'] + 1, 'y': state['y'] + 1}

agents = AgentCollection(MyAgentType(), 10)
model = Model(params, config, update_state_fn, metrics_fn)
model.add_agent_collection('my_agents', agents)
results = model.run()
```

New AgentPy-like Interface:
```python
class MyAgent(jx.Agent):
    def setup(self):
        return {'x': 0, 'y': 0}
    
    def step(self, model_state):
        return {'x': self._state['x'] + 1, 'y': self._state['y'] + 1}

class MyModel(jx.Model):
    def setup(self):
        self.agents = self.add_agents(10, MyAgent)
    
    def step(self):
        # Agents are stepped automatically
        pass

model = MyModel({'steps': 100})
results = model.run()
```

The AgentPy-like interface simplifies the creation of agent-based models while maintaining all the performance benefits of JAX acceleration.