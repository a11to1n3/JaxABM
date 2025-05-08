# JaxABM: JAX-Accelerated Agent-Based Modeling Framework

JaxABM is a high-performance agent-based modeling (ABM) framework that leverages JAX for GPU acceleration, vectorization, and automatic differentiation. This enables significantly faster simulation speeds and advanced capabilities compared to traditional Python-based ABM frameworks.

## Key Features

- **GPU Acceleration**: Run simulations on GPUs with minimal code changes
- **Fully Vectorized**: Uses JAX's vectorization for highly parallel agent simulations
- **Multiple Agent Types**: Support for heterogeneous agent populations
- **Differentiable Simulations**: End-to-end differentiable ABM for gradient-based optimization
- **Powerful Analysis Tools**: Built-in sensitivity analysis and parameter calibration
- **Pure Functional Core**: Leverages JAX's functional programming model
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

## Core Abstractions

The framework is built around several key abstractions:

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

### `Model`

Coordinates the overall simulation:

- `add_agent_collection(name, collection)`: Add an agent collection
- `add_env_state(name, value)`: Add an environmental state variable
- `initialize()`: Prepare the model for simulation
- `step()`: Execute a single time step
- `run(steps)`: Run the full simulation
- `jit_step()`: Get a JIT-compiled step function for maximum performance

## Basic Usage

Here's how to create a simple agent-based model:

```python
import jax
import jax.numpy as jnp
from jax import random

from jaxabm.agent import AgentType, AgentCollection
from jaxabm.core import ModelConfig
from jaxabm.model import Model

# Define an agent type
class RandomWalker(AgentType):
    def __init__(self, init_scale=1.0, step_size=0.1):
        self.init_scale = init_scale
        self.step_size = step_size
        
    def init_state(self, model_config, key):
        return {
            'position': random.normal(key, (2,)) * self.init_scale
        }
    
    def update(self, state, model_state, model_config, key):
        # Move in a random direction
        movement = random.normal(key, (2,)) * self.step_size
        new_position = state['position'] + movement
        
        # Return updated state
        return {'position': new_position}

# Define environment update function
def update_model_state(env_state, agent_states, params, key):
    # Calculate statistics from agent states
    walker_states = agent_states.get('walkers')
    positions = walker_states['position']
    
    mean_pos = jnp.mean(positions, axis=0)
    std_pos = jnp.std(positions, axis=0)
    
    return {
        'mean_position': mean_pos,
        'std_position': std_pos,
    }

# Define metrics function
def compute_metrics(env_state, agent_states, params):
    return {
        'mean_x': env_state['mean_position'][0],
        'mean_y': env_state['mean_position'][1],
        'std_x': env_state['std_position'][0],
        'std_y': env_state['std_position'][1],
    }

# Create agent collection
walkers = AgentCollection(
    agent_type=RandomWalker(init_scale=1.0, step_size=0.1),
    num_agents=1000
)

# Create model
model = Model(
    params={},
    config=ModelConfig(seed=42, steps=100),
    update_state_fn=update_model_state,
    metrics_fn=compute_metrics
)

# Add agents and initial environment state
model.add_agent_collection('walkers', walkers)
model.add_env_state('mean_position', jnp.zeros(2))
model.add_env_state('std_position', jnp.ones(2))

# Run simulation
results = model.run()

# Access results
print(f"Final position: ({results['mean_x'][-1]:.4f}, {results['mean_y'][-1]:.4f})")
print(f"Final std dev: ({results['std_x'][-1]:.4f}, {results['std_y'][-1]:.4f})")
```

## Advanced Features

### Sensitivity Analysis

JaxABM provides tools to analyze how model outputs respond to parameter changes:

```python
from jaxabm.analysis import SensitivityAnalysis

# Create model factory function
def create_model(params=None, config=None):
    # Create model with parameters from the params dict
    propensity_to_consume = params.get('propensity_to_consume', 0.8)
    productivity = params.get('productivity', 1.0)
    
    # Create and return model
    # ...

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

- `examples/minimal_example.py`: Minimal demonstration of the core API
- `examples/jax_abm_simple.py`: Simplified model for quick experimentation
- `examples/jax_abm_example.py`: Detailed economic model with sensitivity analysis
- `examples/jax_abm_professional.py`: Professional example with full analysis toolkit

Run examples with:

```bash
python examples/jax_abm_professional.py --simulation --fast
python examples/jax_abm_professional.py --sensitivity --calibration --fast
```

## Performance

JaxABM provides significant performance improvements:

- **10-100x** faster than pure Python implementations
- **GPU acceleration** with no code changes
- **Parallel agent updates** through vectorization
- **JIT compilation** for optimal performance

## Requirements

- Python 3.8+
- JAX 0.4.1+ (for acceleration features)
- NumPy
- Matplotlib (for visualization)

## License

This project is licensed under the MIT License - see the LICENSE file for details.