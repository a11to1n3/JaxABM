# Advanced Model Calibration Guide

This guide explains the significantly improved calibration system in JaxABM, which provides multiple optimization strategies for automatically tuning model parameters to achieve desired outputs.

## Overview

The improved calibration system includes:

1. **Advanced Gradient-based Methods**: Adam and SGD optimizers with adaptive learning rates
2. **Evolution Strategies (ES)**: Population-based optimization with elite selection
3. **Particle Swarm Optimization (PSO)**: Swarm intelligence approach
4. **Cross-Entropy Method (CEM)**: Distribution-based optimization
5. **Bayesian Optimization**: Gaussian process-based optimization (simplified)
6. **Reinforcement Learning Methods**: Q-learning, Policy Gradient, Actor-Critic, Multi-Agent RL, and DQN
7. **Ensemble Calibration**: Combines multiple methods for robust results

## Key Features

### Robust Evaluation
- Multiple simulation runs for each parameter evaluation
- Confidence intervals for metrics
- Statistical significance testing

### Advanced Loss Functions
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Huber loss (robust to outliers)
- Relative error (percentage-based)

### Smart Convergence
- Early stopping with patience
- Adaptive learning rates
- Gradient clipping for stability

### Comprehensive Monitoring
- Parameter evolution tracking
- Loss history with confidence intervals
- Convergence analysis and visualization

## Usage Examples

### Basic Single-Method Calibration

```python
from jaxabm.analysis import ModelCalibrator

# Define your model factory
def my_model_factory(params, config):
    # Create and return your model instance
    return MyModel(params, config)

# Set up calibration
calibrator = ModelCalibrator(
    model_factory=my_model_factory,
    initial_params={'param1': 1.0, 'param2': 2.0},
    target_metrics={'accuracy': 0.95, 'efficiency': 0.8},
    param_bounds={'param1': (0.1, 5.0), 'param2': (0.5, 10.0)},
    method="adam",  # or "sgd", "es", "pso", "cem", "bayesian"
    loss_type="mse",  # or "mae", "huber", "relative"
    max_iterations=100,
    tolerance=1e-6,
    patience=10
)

# Run calibration
best_params = calibrator.calibrate(verbose=True)
print(f"Optimized parameters: {best_params}")

# Plot results
fig, axes = calibrator.plot_calibration()
```

### Ensemble Calibration

```python
from jaxabm.analysis import EnsembleCalibrator, compare_calibration_methods

# Compare multiple methods
results = compare_calibration_methods(
    model_factory=my_model_factory,
    initial_params={'param1': 1.0, 'param2': 2.0},
    target_metrics={'accuracy': 0.95, 'efficiency': 0.8},
    methods=["adam", "es", "pso", "cem", "q_learning", "policy_gradient"],
    max_iterations=50,
    verbose=True
)

# Get best overall result
best_method = results['best']['method']
best_params = results['best']['params']
best_loss = results['best']['loss']

print(f"Best method: {best_method}")
print(f"Best parameters: {best_params}")
print(f"Final loss: {best_loss}")
```

### Advanced Configuration

```python
# Detailed calibrator setup
calibrator = ModelCalibrator(
    model_factory=my_model_factory,
    initial_params={'learning_rate': 0.01, 'batch_size': 32, 'dropout': 0.1},
    target_metrics={
        'train_accuracy': 0.95,
        'val_accuracy': 0.90,
        'training_time': 100.0  # seconds
    },
    param_bounds={
        'learning_rate': (1e-5, 1e-1),
        'batch_size': (8, 128),
        'dropout': (0.0, 0.5)
    },
    metrics_weights={
        'train_accuracy': 1.0,
        'val_accuracy': 2.0,  # More important
        'training_time': 0.1   # Less important
    },
    method="adam",
    loss_type="huber",  # Robust to outliers
    evaluation_steps=100,
    num_evaluation_runs=5,  # Average over 5 runs
    learning_rate=0.01,
    max_iterations=200,
    tolerance=1e-8,
    patience=20,
    seed=42
)
```

## Optimization Methods Explained

### 1. Adam Optimizer
**Best for**: Smooth, differentiable objective functions
**Advantages**: 
- Adaptive learning rates
- Momentum-based updates
- Good convergence properties
**Use when**: Your model metrics are smooth functions of parameters

```python
calibrator = ModelCalibrator(method="adam", learning_rate=0.01)
```

### 2. Evolution Strategies (ES)
**Best for**: Noisy, non-differentiable objectives
**Advantages**:
- Population-based search
- Robust to noise
- No gradient computation needed
**Use when**: Model evaluation is noisy or discontinuous

```python
calibrator = ModelCalibrator(method="es")
```

### 3. Particle Swarm Optimization (PSO)
**Best for**: Multi-modal optimization landscapes
**Advantages**:
- Good exploration capabilities
- Balances exploration vs exploitation
- Works well with multiple local optima
**Use when**: Parameter space has multiple good solutions

```python
calibrator = ModelCalibrator(method="pso")
```

### 4. Cross-Entropy Method (CEM)
**Best for**: High-dimensional parameter spaces
**Advantages**:
- Efficient sampling
- Good for continuous parameters
- Adaptive distribution updates
**Use when**: You have many parameters to optimize

```python
calibrator = ModelCalibrator(method="cem")
```

### 5. Bayesian Optimization
**Best for**: Expensive function evaluations
**Advantages**:
- Sample efficient
- Uncertainty quantification
- Principled exploration
**Use when**: Each model evaluation is computationally expensive

```python
calibrator = ModelCalibrator(method="bayesian")
```

### 6. Reinforcement Learning Methods

#### Q-Learning
**Best for**: Discrete parameter exploration
**Advantages**:
- Model-free learning
- Exploration-exploitation balance
- Learns optimal parameter policies
**Use when**: You want to learn which parameter changes work best

```python
calibrator = ModelCalibrator(method="q_learning")
```

#### Policy Gradient (REINFORCE)
**Best for**: Continuous parameter spaces
**Advantages**:
- Direct policy optimization
- Handles continuous actions naturally
- Variance reduction with baseline
**Use when**: Parameters are continuous and you want smooth updates

```python
calibrator = ModelCalibrator(method="policy_gradient")
```

#### Actor-Critic
**Best for**: Balanced exploration and stable learning
**Advantages**:
- Combines value and policy learning
- Lower variance than pure policy gradient
- More stable than Q-learning
**Use when**: You want the benefits of both value and policy methods

```python
calibrator = ModelCalibrator(method="actor_critic")
```

#### Multi-Agent RL
**Best for**: Independent parameter control
**Advantages**:
- Each parameter has its own agent
- Parallel parameter exploration
- Handles parameter interactions
**Use when**: Parameters can be optimized somewhat independently

```python
calibrator = ModelCalibrator(method="multi_agent_rl")
```

#### Deep Q-Network (DQN)
**Best for**: Complex parameter relationships
**Advantages**:
- Neural network function approximation
- Experience replay for stability
- Handles high-dimensional parameter spaces
**Use when**: Parameter relationships are complex and non-linear

```python
calibrator = ModelCalibrator(method="dqn")
```

## Loss Function Selection

### Mean Squared Error (MSE)
```python
loss_type="mse"  # Penalizes large errors heavily
```

### Mean Absolute Error (MAE)
```python
loss_type="mae"  # Robust to outliers
```

### Huber Loss
```python
loss_type="huber"  # Combines MSE and MAE benefits
```

### Relative Error
```python
loss_type="relative"  # Percentage-based, good for different scales
```

## Best Practices

### 1. Parameter Bounds
Always specify reasonable bounds for your parameters:
```python
param_bounds = {
    'learning_rate': (1e-6, 1e-1),  # Log scale
    'population_size': (10, 1000),   # Integer-like
    'temperature': (0.1, 10.0)       # Physical parameter
}
```

### 2. Multiple Evaluation Runs
Use multiple runs for robust evaluation:
```python
num_evaluation_runs=5  # Average over 5 random seeds
```

### 3. Appropriate Loss Functions
Choose loss functions based on your metrics:
- Use `"relative"` for metrics with different scales
- Use `"huber"` for noisy environments
- Use `"mse"` for smooth, well-behaved metrics

### 4. Method Selection
- Start with `"adam"` for smooth problems
- Use `"es"` or `"pso"` for noisy/discontinuous problems
- Try ensemble calibration for critical applications

### 5. Convergence Monitoring
```python
tolerance=1e-6,    # Stop when improvement is small
patience=10,       # Stop after 10 iterations without improvement
max_iterations=100 # Maximum iterations
```

## Troubleshooting

### Slow Convergence
- Increase `learning_rate` for gradient methods
- Increase population size for evolutionary methods
- Check if parameter bounds are too restrictive

### Unstable Training
- Decrease `learning_rate`
- Use `"huber"` loss instead of `"mse"`
- Increase `num_evaluation_runs` for more stable estimates

### Poor Final Results
- Try different optimization methods
- Check parameter bounds
- Verify target metrics are achievable
- Use ensemble calibration

### Memory Issues
- Reduce `evaluation_steps`
- Reduce population sizes for evolutionary methods
- Use fewer `num_evaluation_runs`

## Advanced Features

### Custom Model Factories
Your model factory should accept parameters and config:
```python
def my_model_factory(params, config):
    # Use params to configure your model
    model = MyABM(
        learning_rate=params['learning_rate'],
        population_size=int(params['population_size']),
        seed=config.seed
    )
    return model
```

### Metric Weighting
Weight different metrics by importance:
```python
metrics_weights = {
    'accuracy': 2.0,      # Most important
    'speed': 1.0,         # Moderately important  
    'memory_usage': 0.5   # Less important
}
```

### Confidence Intervals
Access confidence intervals for metrics:
```python
history = calibrator.get_calibration_history()
final_ci = history['confidence_intervals'][-1]
for metric, (lower, upper) in final_ci.items():
    print(f"{metric}: {(lower+upper)/2:.3f} ± {(upper-lower)/2:.3f}")
```

## Performance Tips

1. **Start Simple**: Begin with Adam optimizer and basic settings
2. **Use Ensemble**: For critical applications, compare multiple methods
3. **Monitor Progress**: Use `verbose=True` to track optimization
4. **Parallel Evaluation**: Consider parallelizing model evaluations
5. **Warm Starting**: Use results from one method to initialize another

## Example: Complete Calibration Workflow

```python
from jaxabm.analysis import ModelCalibrator, compare_calibration_methods

# 1. Define model factory
def epidemic_model_factory(params, config):
    return EpidemicModel(
        infection_rate=params['infection_rate'],
        recovery_rate=params['recovery_rate'],
        population_size=1000,
        seed=config.seed
    )

# 2. Set targets based on real data
target_metrics = {
    'peak_infections': 250,
    'total_infected': 600,
    'epidemic_duration': 50
}

# 3. Define parameter space
initial_params = {'infection_rate': 0.1, 'recovery_rate': 0.05}
param_bounds = {
    'infection_rate': (0.01, 0.5),
    'recovery_rate': (0.01, 0.2)
}

# 4. Run ensemble calibration
results = compare_calibration_methods(
    model_factory=epidemic_model_factory,
    initial_params=initial_params,
    target_metrics=target_metrics,
    methods=["adam", "es", "pso"],
    max_iterations=50
)

# 5. Use best parameters
best_params = results['best']['params']
print(f"Calibrated parameters: {best_params}")

# 6. Validate results
final_model = epidemic_model_factory(best_params, ModelConfig(seed=123))
validation_results = final_model.run(steps=100)
print(f"Validation metrics: {validation_results}")
```

This improved calibration system provides a robust, flexible framework for automatically tuning your agent-based models to match empirical data or achieve desired behaviors. 