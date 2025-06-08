# JaxABM: JAX-Accelerated Agent-Based Modeling Framework

JaxABM is a high-performance agent-based modeling (ABM) framework that leverages JAX for GPU acceleration, vectorization, and automatic differentiation, now with an easy-to-use AgentPy-like interface. This enables significantly faster simulation speeds and advanced capabilities compared to traditional Python-based ABM frameworks.

## Key Features

- **Easy-to-use Interface**: AgentPy-like API for intuitive model development
- **GPU Acceleration**: Run simulations on GPUs with minimal code changes
- **Fully Vectorized**: Uses JAX's vectorization for highly parallel agent simulations
- **Multiple Agent Types**: Support for heterogeneous agent populations
- **Differentiable Simulations**: End-to-end differentiable ABM for gradient-based optimization
- **Powerful Analysis Tools**: Built-in sensitivity analysis and parameter calibration
- **Spatial Structures**: Built-in support for grid and network environments
- **Backward Compatible**: Legacy API support for traditional (non-JAX) modeling

## Installation

### Basic Installation

```bash
pip install jaxabm
```

### Install with JAX capabilities

First install JAX following the [official instructions](https://github.com/google/jax#installation) (for GPU support), then:

```bash
pip install jaxabm[jax]
```

## Quick Start

Here's a simple example of a model with agents that move randomly:

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
        # Get current position
        x = self._state['x']
        y = self._state['y']
        
        # Move randomly (using a simple deterministic rule for this example)
        x += 0.01
        y += 0.01
        
        # Wrap around at boundaries
        x = x % 1.0
        y = y % 1.0
        
        # Return updated state
        return {
            'x': x,
            'y': y
        }

class MyModel(jx.Model):
    def setup(self):
        """Set up model with agents and environment."""
        # Add agents
        self.agents = self.add_agents(10, MyAgent)
        
        # Set up environment
        self.env.add_state('time', 0)
    
    def step(self):
        """Execute model logic each step."""
        # Update environment time
        # Note: Agents are updated automatically by the framework
        if hasattr(self._jax_model, 'state'):
            time = self._jax_model.state['env'].get('time', 0)
            self._jax_model.add_env_state('time', time + 1)
        
        # Record data
        self.record('time', time)

# Run model
model = MyModel({'steps': 100})
results = model.run()

# Plot results
results.plot()
```

## The AgentPy-like Interface

JaxABM now provides an easy-to-use, AgentPy-like interface built on top of the high-performance JAX core.

### Agent

The `Agent` class is the base class for all agents in the model. To create a custom agent, inherit from this class and override the `setup` and `step` methods.

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
        return {
            'x': self._state['x'] + 0.1,
            'y': self._state['y'] + 0.1
        }
```

### AgentList

The `AgentList` class is a container for managing collections of agents.

```python
# In Model.setup():
self.agents = self.add_agents(10, MyAgent)

# Access agent attributes:
x_positions = self.agents.x  # Returns array of x values

# Filter agents:
active_agents = self.agents.select(lambda agents: agents.active)
```

### Environment

The `Environment` class is a container for environment state and methods for creating and managing spatial structures.

```python
# In Model.setup():
self.env.add_state('temperature', 25.0)

# Access environment state:
temp = self.env.temperature
```

### Grid and Network

For spatial models, the `Grid` and `Network` classes provide structures for agent interactions.

```python
# Create a grid:
self.grid = jx.Grid(self, (10, 10))

# Position agents on grid:
self.grid.position_agents(self.agents)

# Create a network:
self.network = jx.Network(self)

# Add edges:
self.network.add_edge(agent1, agent2)
```

### Model

The `Model` class is the base class for all models. It provides methods for setting up, running, and analyzing models.

```python
class MyModel(jx.Model):
    def setup(self):
        """Set up model with agents and environment."""
        self.agents = self.add_agents(10, MyAgent)
        self.env.add_state('time', 0)
    
    def step(self):
        """Execute model logic each step."""
        # Environment updates (agent updates happen automatically)
        if hasattr(self._jax_model, 'state'):
            time = self._jax_model.state['env'].get('time', 0)
            self._jax_model.add_env_state('time', time + 1)
        
        # Record data
        self.record('time', time)
    
    def end(self):
        """Execute code at the end of a simulation."""
        print("Simulation completed!")

# Create and run model
model = MyModel({'steps': 100})
results = model.run()
```

### Results

The `Results` class is a container for simulation results. It provides methods for accessing and visualizing results.

```python
# Run model and get results
results = model.run()

# Plot all metrics
results.plot()

# Access specific variables
results.variables.agent.x.plot()

# Save results
results.save('my_results.pkl')

# Load results
results = jx.Results.load('my_results.pkl')
```

## Advanced Features

### Sensitivity Analysis

JaxABM provides tools to analyze how model outputs respond to parameter changes:

```python
from jaxabm.analysis import SensitivityAnalysis

# Create model factory function
def create_model(params=None, config=None):
    # Create model with parameters from the params dict
    model = MyModel(params)
    return model

# Perform sensitivity analysis
sensitivity = SensitivityAnalysis(
    model_factory=create_model,
    param_ranges={
        'propensity_to_consume': (0.6, 0.9),
        'productivity': (0.5, 1.5),
    },
    metrics_of_interest=['gdp', 'unemployment', 'inequality'],
    num_samples=10
)

# Run analysis
results = sensitivity.run()

# Calculate sensitivity indices
indices = sensitivity.sobol_indices()
```

### Model Calibration

Find optimal parameters to match target metrics using gradient-based or RL-based methods:

```python
from jaxabm.analysis import ModelCalibrator

# Define target metrics
target_metrics = {
    'gdp': 10.0,
    'unemployment': 0.05,
    'inequality': 2.0
}

# Initialize calibrator
calibrator = ModelCalibrator(
    model_factory=create_model,
    initial_params={
        'propensity_to_consume': 0.7,
        'productivity': 1.0
    },
    target_metrics=target_metrics,
    metrics_weights={
        'gdp': 0.1, 
        'unemployment': 1.0,
        'inequality': 0.5
    },
    learning_rate=0.01,
    max_iterations=20,
    method='gradient'  # or 'rl'
)

# Run calibration
optimal_params = calibrator.calibrate()
```

## Examples

The package includes several example models demonstrating different features:

- `examples/random_walk.py`: Simple model with random walking agents
- `examples/schelling_model.py`: Classic Schelling segregation model
- `examples/minimal_example_agentpy.py`: AgentPy-like version of the minimal example
- `examples/agentpy_interface_example.py`: Bouncing agents with AgentPy-like interface
- `examples/minimal_example.py`: Original JaxABM API example
- `examples/jax_abm_simple.py`: Simplified model with original API
- `examples/jax_abm_example.py`: Detailed economic model with sensitivity analysis

Run examples with:

```bash
python examples/random_walk.py
python examples/schelling_model.py
```

## Core Abstractions (Original API)

The framework is also built around several key core abstractions that power the AgentPy-like interface:

### `AgentType` Protocol

Defines the behavior of agents:

- `init_state(model_config, key)`: Initialize agent state
- `update(state, model_state, model_config, key)`: Update agent state based on current state and environment

### `AgentCollection`

Manages a collection of agents of the same type:

- `__init__(agent_type, num_agents)`: Create collection placeholder
- `init(key, model_config)`: Initialize all agents in the collection
- `update(model_state, key, model_config)`: Update all agents in parallel
- `states`: Access the current states of all agents
- `filter(condition)`: Creates a filtered subset of agents

### `ModelConfig` 

Provides simulation configuration:

- `seed`: Random seed for reproducibility
- `steps`: Number of simulation steps
- `track_history`: Whether to track model history
- `collect_interval`: Interval for collecting metrics

### `JaxModel`

Coordinates the overall simulation:

- `add_agent_collection(name, collection)`: Add an agent collection
- `add_env_state(name, value)`: Add an environmental state variable
- `initialize()`: Prepare the model for simulation
- `step()`: Execute a single time step
- `run(steps)`: Run the full simulation
- `jit_step()`: Get a JIT-compiled step function for maximum performance

## Performance

JaxABM provides significant performance improvements:

- **10-100x** faster than pure Python implementations
- **GPU acceleration** with no code changes
- **Parallel agent updates** through vectorization
- **JIT compilation** for optimal performance

## Citation

If you use JaxABM in your research, please cite:

### BibTeX

```bibtex
@software{pham2025jaxabm,
  title={JaxABM: JAX-Accelerated Agent-Based Modeling Framework},
  author={Pham, Anh-Duy and D'Orazio, Paola},
  year={2025},
  month={June},
  version={0.1.0},
  url={https://github.com/a11to1n3/JaxABM},
  note={High-performance agent-based modeling framework with GPU acceleration and reinforcement learning calibration}
}
```

### APA Style

Pham, A.-D., & D'Orazio, P. (2025). *JaxABM: JAX-Accelerated Agent-Based Modeling Framework* (Version 0.1.0) [Computer software]. https://github.com/a11to1n3/JaxABM

### IEEE Style

A.-D. Pham and P. D'Orazio, "JaxABM: JAX-Accelerated Agent-Based Modeling Framework," Version 0.1.0, June 2025. [Online]. Available: https://github.com/a11to1n3/JaxABM

### Key Features to Cite

When citing JaxABM, you may want to highlight these innovations:

- **GPU-accelerated agent-based modeling** with JAX backend
- **Advanced reinforcement learning calibration methods** (Actor-Critic, Policy Gradient, Q-Learning, DQN)
- **High-performance vectorized simulations** with 10-100x speedup over traditional ABM frameworks
- **Differentiable agent-based models** enabling gradient-based optimization
- **Comprehensive parameter optimization toolkit** with multiple calibration algorithms

### Related Publications

If you use specific features, consider citing the underlying methodologies:

- For **reinforcement learning calibration**: Reference the specific RL algorithms used (Actor-Critic, Policy Gradient, etc.)
- For **sensitivity analysis**: Sobol indices methodology
- For **JAX backend**: The JAX library for high-performance machine learning research

## Requirements

- Python 3.8+
- JAX 0.4.1+ (for acceleration features)
- NumPy
- Matplotlib (for visualization)

## License

This project is licensed under the MIT License - see the LICENSE file for details.