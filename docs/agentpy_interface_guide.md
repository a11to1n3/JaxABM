# JaxABM with AgentPy-like Interface

JaxABM now features an AgentPy-like interface for easier model development while maintaining all the performance benefits of JAX acceleration.

## Overview

The new interface provides a more intuitive, object-oriented approach to building agent-based models, similar to the popular AgentPy framework. The interface includes:

- `Agent` class for creating agents with `setup` and `step` methods
- `AgentList` for managing collections of agents
- `Environment` for managing environment state
- `Grid` and `Network` for spatial structures
- `Model` class with `setup`, `step`, and `run` methods
- `Results` class for analyzing and visualizing results
- `Parameter`, `SensitivityAnalyzer`, and `ModelCalibrator` for parameter analysis and optimization

## How to Use

### Agents

Create agents by inheriting from the `Agent` class and implementing `setup` and `step` methods:

```python
import jaxabm as jx
import jax.numpy as jnp

class MyAgent(jx.Agent):
    def setup(self):
        """Initialize agent state."""
        return {
            'x': 0.5,
            'y': 0.5
        }
    
    def step(self, model_state):
        """Update agent state."""
        # Get current state
        x = self._state['x']
        y = self._state['y']
        
        # Move in a deterministic pattern
        x = (x + 0.01) % 1.0
        y = (y + 0.01) % 1.0
        
        # Return updated state
        return {
            'x': x,
            'y': y
        }
```

### Models

Create models by inheriting from the `Model` class and implementing `setup` and `step` methods:

```python
class MyModel(jx.Model):
    def setup(self):
        """Set up model with agents and environment."""
        # Add agents
        self.agents = self.add_agents(10, MyAgent)
        
        # Set up environment
        self.env.add_state('time', 0)
    
    def step(self):
        """Execute model logic each step."""
        # Update environment (agents are updated automatically)
        self.env.add_state('time', self.env.time + 1)
        
        # Record data
        self.record('time', self.env.time)
    
    def end(self):
        """Execute at the end of simulation."""
        print("Simulation completed!")
```

### Running Models

Run models and analyze results:

```python
# Create model with parameters
model = MyModel({
    'steps': 100,
    'seed': 42
})

# Run simulation
results = model.run()

# Plot results
results.plot()

# Access specific variables
results.variables.agent.x.plot()
```

### Sensitivity Analysis

Perform sensitivity analysis to understand how parameters affect model outcomes:

```python
# Define parameters
growth_rate = jx.Parameter('growth_rate', bounds=(0.01, 0.1))
initial_pop = jx.Parameter('initial_population', bounds=(10, 100))

# Create analyzer
analyzer = jx.SensitivityAnalyzer(
    model_class=MyModel,
    parameters=[growth_rate, initial_pop],
    n_samples=10,
    metrics=['population', 'resources']
)

# Run analysis
results = analyzer.run()

# Calculate sensitivity
sensitivity = analyzer.calculate_sensitivity()

# Plot sensitivity
analyzer.plot('population')
```

### Model Calibration

Calibrate model parameters to match target metrics:

```python
# Define parameters
growth_rate = jx.Parameter('growth_rate', bounds=(0.01, 0.1))
initial_pop = jx.Parameter('initial_population', bounds=(10, 100))

# Define target metrics
target_metrics = {
    'population': 50,
    'resources': 200
}

# Create calibrator
calibrator = jx.ModelCalibrator(
    model_class=MyModel,
    parameters=[growth_rate, initial_pop],
    target_metrics=target_metrics,
    metrics_weights={'population': 1.0, 'resources': 0.5},
    max_iterations=20
)

# Run calibration
optimal_params = calibrator.run()

# Plot calibration progress
calibrator.plot_progress()
```

## Examples

The framework includes several examples demonstrating the AgentPy-like interface:

- `examples/simple_model.py`: A minimal model to demonstrate the basic functionality
- `examples/random_walk.py`: A more complex model with random walking agents
- `examples/schelling_model.py`: Classic Schelling segregation model
- `examples/minimal_example_agentpy.py`: AgentPy-like version of the minimal example
- `examples/sensitivity_calibration_example.py`: Example of sensitivity analysis and model calibration

## Key Differences from AgentPy

While the interface is similar to AgentPy, there are some differences to be aware of:

1. Agent state is represented as dictionaries returned from `setup` and `step` methods
2. JAX's constraints on mutation and randomness apply
3. Environment state is updated using `env.add_state(name, value)` instead of direct attribute assignment
4. The performance benefits come from JAX, which requires functional programming patterns

## Migrating from Original JaxABM API

If you're using the original JaxABM API, here's how to migrate:

| Original API | AgentPy-like API |
|--------------|------------------|
| `AgentType` protocol | `Agent` class |
| `init_state` method | `setup` method |
| `update` method | `step` method |
| `AgentCollection` | `AgentList` |
| `model.add_agent_collection()` | `model.add_agents()` |
| Custom update function | `Model.step()` method |
| Custom metrics function | `Model.compute_metrics()` method |
| `SensitivityAnalysis` | `SensitivityAnalyzer` |
| `ModelCalibrator` | `ModelCalibrator` |

## Performance Considerations

The AgentPy-like interface is built on top of the same high-performance JAX core as the original API, so you can expect the same speed benefits. However, there are a few things to keep in mind:

1. Use JAX-compatible operations in agent and model methods
2. Avoid Python loops and conditionals in performance-critical code
3. Use JAX's functional programming patterns for best performance

For more information, see the [JAX documentation](https://jax.readthedocs.io/).