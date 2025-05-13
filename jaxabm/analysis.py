"""
Analysis module for JAX-based agent-based modeling.

This module provides tools for analyzing and calibrating agent-based models
built with the jaxabm framework, including sensitivity analysis and 
parameter optimization techniques that leverage JAX's capabilities.
"""

from typing import Any, Dict, List, Optional, Tuple, Callable, Union, TypeVar
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random

# Import Model from .model and ModelConfig from .core
from .model import Model
from .core import ModelConfig

# Type variables for better type annotations
ModelFactory = TypeVar('ModelFactory', bound=Callable[..., Model])
PRNGKey = jax.Array


class SensitivityAnalysis:
    """Perform sensitivity analysis on model parameters.
    
    This class provides tools for analyzing how changes in model parameters
    affect model outputs, using efficient sampling techniques and 
    sensitivity indices calculation.
    
    Attributes:
        model_factory: Function to create model instances
        param_ranges: Dictionary mapping parameter names to (min, max) ranges
        metrics_of_interest: List of metric names to analyze
        num_samples: Number of parameter samples to generate
        key: JAX random key
        samples: Generated parameter samples
        results: Analysis results (populated after run())
    """
    
    def __init__(
        self,
        model_factory: ModelFactory,
        param_ranges: Dict[str, Tuple[float, float]],
        metrics_of_interest: List[str],
        num_samples: int = 100,
        seed: int = 0
    ):
        """Initialize sensitivity analysis.
        
        Args:
            model_factory: Function to create model instances
            param_ranges: Dictionary mapping parameter names to (min, max) ranges
            metrics_of_interest: List of metric names to analyze
            num_samples: Number of parameter samples to generate
            seed: Random seed
        """
        self.model_factory = model_factory
        self.param_ranges = param_ranges
        self.metrics_of_interest = metrics_of_interest
        self.num_samples = num_samples
        self.key = random.PRNGKey(seed)
        
        # Generate samples using Latin Hypercube Sampling
        self.samples = self._generate_lhs_samples()
        self.results = None
    
    def _generate_lhs_samples(self) -> jax.Array:
        """Generate Latin Hypercube Samples for parameters.
        
        Latin Hypercube Sampling ensures better coverage of the parameter space
        than simple random sampling.
        
        Returns:
            Array of shape (num_samples, num_parameters) with sampled parameter values
        """
        self.key, subkey = random.split(self.key)
        
        # Create normalized LHS samples (0-1)
        n_params = len(self.param_ranges)
        points = jnp.linspace(0, 1, self.num_samples + 1)[:-1]  # n points in [0, 1)
        points = points + random.uniform(subkey, (self.num_samples,)) / self.num_samples  # Add jitter
        
        # Create a permutation of these points for each parameter
        samples = jnp.zeros((self.num_samples, n_params))
        for i, param in enumerate(self.param_ranges):
            self.key, subkey = random.split(self.key)
            perm = random.permutation(subkey, points)
            samples = samples.at[:, i].set(perm)
        
        # Scale samples to parameter ranges
        for i, (param, (min_val, max_val)) in enumerate(self.param_ranges.items()):
            samples = samples.at[:, i].multiply(max_val - min_val)
            samples = samples.at[:, i].add(min_val)
        
        return samples
    
    def run(self, verbose: bool = True) -> Dict[str, jax.Array]:
        """Run sensitivity analysis.
        
        Args:
            verbose: Whether to print progress information
            
        Returns:
            Dictionary mapping metric names to arrays of results
        """
        if verbose:
            print(f"Running sensitivity analysis with {self.num_samples} samples...")
            
        param_names = list(self.param_ranges.keys())
        metrics_results = {metric: jnp.zeros(self.num_samples) for metric in self.metrics_of_interest}
        
        # Run model for each parameter sample
        for i in range(self.num_samples):
            if verbose:
                print(f"\nSample {i+1}/{self.num_samples}")
            
            # Construct parameter dictionary for this sample
            params = {param: float(self.samples[i, j]) for j, param in enumerate(param_names)}
            if verbose:
                print(f"Parameters: {', '.join([f'{k}={v:.4f}' for k, v in params.items()])}")
            
            # Create and run model using the factory
            # Pass parameters and create a config with the specific seed
            seed_value = i + 1000  # Use the sample index for reproducibility
            config = ModelConfig(seed=seed_value) 
            # Assuming model_factory signature is factory(params=..., config=...)
            # The factory itself needs to handle adding agents/state.
            model = self.model_factory(params=params, config=config)
            
            if verbose:
                print("Running model...")
                
            # model.run() now handles initialization internally
            results = model.run()
            
            # Extract metrics of interest
            if verbose:
                print("Results:")
                
            # Handle both dictionary and Results objects
            if hasattr(results, '_data'):
                results_dict = results._data
            else:
                results_dict = results
                
            for metric in self.metrics_of_interest:
                if metric in results_dict and results_dict[metric]:
                    # Store the final value of the metric
                    value = results_dict[metric][-1]
                    metrics_results[metric] = metrics_results[metric].at[i].set(value)
                    if verbose:
                        print(f"  {metric}: {float(value):.4f}")
        
        self.results = metrics_results
        if verbose:
            print("\nSensitivity analysis complete!")
            
        return metrics_results
    
    def sobol_indices(self) -> Dict[str, Dict[str, float]]:
        """Calculate sensitivity indices for each parameter and metric.
        
        This is a simplified implementation that calculates correlation-based
        indices as a proxy for Sobol indices. For a full Sobol analysis,
        specialized sampling would be required.
        
        Returns:
            Dictionary mapping metric names to dictionaries of parameter name -> sensitivity index
        """
        # NOTE: This method calculates squared correlation coefficients as a simplified 
        # proxy for true Sobol indices. A full Sobol analysis would require 
        # different sampling techniques (e.g., Saltelli sampling).
        if self.results is None:
            raise ValueError("Must run sensitivity analysis before calculating indices")
        
        param_names = list(self.param_ranges.keys())
        indices = {}
        
        for metric, values in self.results.items():
            # Normalize the metric values
            values_norm = (values - jnp.mean(values)) / (jnp.std(values) + 1e-8)
            
            # Calculate correlation coefficients as a simple sensitivity measure
            metric_indices = {}
            for i, param in enumerate(param_names):
                # Use correlation coefficient as a simple proxy for sensitivity
                param_values = self.samples[:, i]
                param_values_norm = (param_values - jnp.mean(param_values)) / (jnp.std(param_values) + 1e-8)
                
                # Calculate correlation coefficient
                corr = jnp.mean(param_values_norm * values_norm)
                metric_indices[param] = float(corr ** 2)  # Square to get something like an R² value
            
            indices[metric] = metric_indices
        
        return indices

    def plot(self, metric=None, ax=None, **kwargs):
        """Plot sensitivity analysis results.
        
        Args:
            metric: Metric to plot. If None, plot sobol indices for all metrics.
            ax: Matplotlib axis to use for plotting.
            **kwargs: Additional keyword arguments to pass to plotting function.
            
        Returns:
            Matplotlib axis.
        """
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
        
        # Get sobol indices
        indices = self.sobol_indices()
        
        # Choose metric to plot
        if metric is None and self.metrics_of_interest:
            metric = self.metrics_of_interest[0]
        
        if metric in indices:
            # Get indices for this metric
            metric_indices = indices[metric]
            
            # Sort indices by value
            sorted_indices = sorted(metric_indices.items(), key=lambda x: x[1], reverse=True)
            
            # Plot bar chart
            params = [p for p, _ in sorted_indices]
            values = [v for _, v in sorted_indices]
            ax.bar(params, values, **kwargs)
            ax.set_xlabel('Parameter')
            ax.set_ylabel('Sensitivity Index')
            ax.set_title(f'Sensitivity Indices for {metric}')
            
            # Rotate x-labels if there are many parameters
            if len(params) > 3:
                import matplotlib.pyplot as plt
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        return ax
    
    def plot_indices(self, figsize: Tuple[int, int] = (10, 6)) -> Any:
        """Plot the sensitivity indices.
        
        Args:
            figsize: Figure size as (width, height)
            
        Returns:
            Matplotlib figure and axes
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            raise ImportError("Matplotlib is required for plotting. Install it with 'pip install matplotlib'")
        
        indices = self.sobol_indices()
        
        metrics = list(indices.keys())
        params = list(indices[metrics[0]].keys())
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(metrics))
        width = 0.8 / len(params)
        
        for i, param in enumerate(params):
            param_values = [indices[metric][param] for metric in metrics]
            offset = width * i - width * len(params) / 2 + width / 2
            ax.bar(x + offset, param_values, width, label=param)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Sensitivity Index')
        ax.set_title('Parameter Sensitivity Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend(loc='best')
        
        plt.tight_layout()
        return fig, ax


class ModelCalibrator:
    """Calibrate model parameters using gradient-based optimization or RL.
    
    This class provides methods for automatically tuning model parameters
    to achieve desired outputs, using either gradient-based optimization
    (leveraging JAX's autodiff) or reinforcement learning approaches.
    
    Attributes:
        model_factory: Function to create model instances
        params: Current parameter values
        target_metrics: Target values for each metric
        metrics_weights: Importance weights for each metric in the loss function
        learning_rate: Learning rate for optimization
        max_iterations: Maximum number of optimization iterations
        method: Calibration method ("gradient" or "rl")
        loss_history: History of loss values during calibration
        param_history: History of parameter values during calibration
    """
    
    def __init__(
        self, 
        model_factory: ModelFactory,
        initial_params: Dict[str, float],
        target_metrics: Dict[str, float],
        metrics_weights: Optional[Dict[str, float]] = None,
        learning_rate: float = 0.01,
        max_iterations: int = 100,
        method: str = "gradient",
    ):
        """Initialize model calibrator.
        
        Args:
            model_factory: Function to create model instances
            initial_params: Initial parameter values
            target_metrics: Target metric values
            metrics_weights: Weights for each metric in the loss function
            learning_rate: Learning rate for optimization
            max_iterations: Maximum number of optimization iterations
            method: Calibration method ("gradient" or "rl")
        """
        self.model_factory = model_factory
        self.params = initial_params
        self.target_metrics = target_metrics
        self.metrics_weights = metrics_weights or {k: 1.0 for k in target_metrics}
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.method = method
        self.loss_history = []
        self.param_history = []
        
        # Initialize random key
        self.key = random.PRNGKey(0)
        
        # Set up the appropriate calibration function
        if method == "gradient":
            self._setup_gradient_calibration()
        elif method == "rl":
            self._setup_rl_calibration()
        else:
            raise ValueError(f"Unknown calibration method: {method}")
    
    def _setup_gradient_calibration(self):
        """Set up gradient-based calibration.
        
        This method defines the loss function and creates a gradient function
        using JAX's automatic differentiation.
        """
        # Define loss function
        def loss_fn(params_flat):
            # Reshape flat parameters back to dictionary
            params = {}
            idx = 0
            for name, value in self.params.items():
                params[name] = params_flat[idx]
                idx += 1
            
            # Create model using factory - use a fixed seed for gradient calculation
            # Pass parameters and create a config with the fixed seed
            seed_value = 42 # Fixed seed for reproducibility within gradient calculation
            config = ModelConfig(seed=seed_value) 
            # Assuming model_factory signature is factory(params=..., config=...)
            model = self.model_factory(params=params, config=config)
            
            # Run the model for a small number of steps - calibration models should use few steps
            steps = 10  # Use a small default number of steps for calibration
            metrics_dict = model.run(steps=steps)
            
            # Get the final metrics
            metrics = {metric: values[-1] if values else 0.0 for metric, values in metrics_dict.items()}
            
            # Check if metrics are available
            if not metrics or not isinstance(metrics, dict):
                return jnp.array(0.0)  # No metrics collected
            
            # Calculate weighted loss directly with JAX operations
            loss = jnp.array(0.0)
            for metric, target in self.target_metrics.items():
                if metric in metrics:
                    value = metrics[metric]
                    weight = self.metrics_weights[metric]
                    loss += weight * ((value - target) ** 2)
            
            return loss
        
        # Create gradient function
        self.grad_fn = jit(grad(loss_fn))
        
    def _setup_rl_calibration(self):
        """Set up reinforcement learning-based calibration.
        
        This method defines the policy network and reward function for RL-based
        parameter tuning.
        """
        # NOTE: The current RL implementation uses a very basic policy network
        # (parameter perturbation) and is intended as a placeholder. 
        # A more sophisticated implementation would typically involve a neural network.

        # Define policy network (simplified version)
        def policy_network(params_flat, key):
            # Apply a simple transformation to parameters
            # In a real implementation, this would be a neural network
            noise = random.normal(key, shape=params_flat.shape) * 0.1
            return params_flat + noise
        
        # Define reward function
        def reward_fn(params_flat):
            # Reshape flat parameters back to dictionary
            params = {}
            idx = 0
            for name, value in self.params.items():
                params[name] = params_flat[idx]
                idx += 1
            
            # Create model using factory - use a fixed seed for reward calculation
            # Pass parameters and create a config with the fixed seed
            seed_value = 42 # Fixed seed for reproducibility
            config = ModelConfig(seed=seed_value) 
            # Assuming model_factory signature is factory(params=..., config=...)
            model = self.model_factory(params=params, config=config)
            
            # Run the model for a small number of steps - calibration models should use few steps
            steps = 10  # Use a small default number of steps for calibration
            metrics_dict = model.run(steps=steps)
            
            # Get the final metrics
            metrics = {metric: values[-1] if values else 0.0 for metric, values in metrics_dict.items()}
            
            # Check if metrics are available
            if not metrics or not isinstance(metrics, dict):
                return jnp.array(0.0)  # No metrics collected
            
            # Calculate reward (negative loss) directly with JAX operations
            reward = jnp.array(0.0)
            for metric, target in self.target_metrics.items():
                if metric in metrics:
                    value = metrics[metric]
                    weight = self.metrics_weights[metric]
                    reward -= weight * ((value - target) ** 2)
            
            return reward
        
        self.policy_network = policy_network
        self.reward_fn = reward_fn
    
    def calibrate(self, verbose: bool = True, param_lower_bound: float = 0.01, param_upper_bound: float = 10.0) -> Dict[str, float]:
        """Run calibration process and return optimized parameters.
        
        Args:
            verbose: Whether to print progress information
            param_lower_bound: Optional lower bound for parameter clipping during gradient descent.
            param_upper_bound: Optional upper bound for parameter clipping during gradient descent.
            
        Returns:
            Dictionary of optimized parameter values
        """
        if verbose:
            print("Starting calibration process...")
            
        # Flatten parameters for optimization
        param_names = list(self.params.keys())
        params_flat = jnp.array([self.params[name] for name in param_names])
        
        self.param_history.append(dict(self.params))
        
        if self.method == "gradient":
            if verbose:
                print(f"Using gradient-based optimization with {self.max_iterations} iterations")
                
            for i in range(self.max_iterations):
                if verbose:
                    print(f"\nIteration {i+1}/{self.max_iterations}")
                    print(f"Current parameters: {', '.join([f'{k}={v:.4f}' for k, v in self.params.items()])}")
                
                # Calculate gradient
                if verbose:
                    print("Computing gradient...")
                    
                grads = self.grad_fn(params_flat)
                
                # Apply gradient clipping to prevent extreme values
                grad_norm = jnp.linalg.norm(grads)
                if verbose:
                    print(f"Gradient norm: {float(grad_norm):.6f}")
                    
                if jnp.isnan(grad_norm) or grad_norm > 10.0:
                    if verbose:
                        print("Gradient too large, applying clipping...")
                    clip_value = 1.0
                    grads = jnp.clip(grads, -clip_value, clip_value)
                
                # Update parameters
                params_flat = params_flat - self.learning_rate * grads
                
                # Clip parameters to bounds
                params_flat = jnp.clip(params_flat, param_lower_bound, param_upper_bound)
                
                # Update parameter dictionary
                for j, name in enumerate(param_names):
                    self.params[name] = float(params_flat[j])
                
                # Record history
                self.param_history.append(dict(self.params))
                
                # Calculate and record loss
                if verbose:
                    print("Evaluating new parameters...")
                    
                # Create model instance for evaluation with a random seed
                self.key, subkey = random.split(self.key)
                seed_value = random.randint(subkey, (), 0, 1_000_000)
                config = ModelConfig(seed=seed_value.item())
                model = self.model_factory(params=self.params, config=config)
                
                # Run the model for evaluation
                results = model.run(steps=10)
                
                loss = 0.0
                for metric, target in self.target_metrics.items():
                    if metric in results and results[metric]:
                        value = results[metric][-1]
                        weight = self.metrics_weights[metric]
                        metric_loss = weight * ((value - target) ** 2)
                        loss += metric_loss
                        if verbose:
                            print(f"  Metric '{metric}': target={target:.4f}, current={value:.4f}, loss={float(metric_loss):.6f}")
                
                self.loss_history.append(float(loss))
                if verbose:
                    print(f"Current loss: {float(loss):.6f}")
                
                # Early stopping if loss is small enough
                if loss < 1e-6:
                    if verbose:
                        print("Early stopping: loss below threshold")
                    break
                
        elif self.method == "rl":
            if verbose:
                print(f"Using RL-based optimization with {self.max_iterations} iterations")
                
            for i in range(self.max_iterations):
                if verbose:
                    print(f"\nIteration {i+1}/{self.max_iterations}")
                    print(f"Current parameters: {', '.join([f'{k}={v:.4f}' for k, v in self.params.items()])}")
                
                # Generate key for policy
                self.key, policy_key = random.split(self.key)
                
                # Sample parameters from policy
                if verbose:
                    print("Sampling new parameters from policy...")
                    
                new_params_flat = self.policy_network(params_flat, policy_key)
                
                # Calculate reward
                if verbose:
                    print("Evaluating sampled parameters...")
                    
                reward = self.reward_fn(new_params_flat)
                
                # Update parameters if reward improved
                current_reward = self.reward_fn(params_flat)
                if reward > current_reward:
                    if verbose:
                        print("Parameters improved, updating...")
                        
                    params_flat = new_params_flat
                    
                    # Update parameter dictionary
                    for j, name in enumerate(param_names):
                        self.params[name] = float(params_flat[j])
                elif verbose:
                    print("Parameters did not improve, keeping current values")
                
                # Record history
                self.param_history.append(dict(self.params))
                self.loss_history.append(float(-reward))  # Convert reward to loss
                
                # Report current values vs targets
                if verbose:
                    for metric, target in self.target_metrics.items():
                        # Create a model to check current metrics
                        self.key, subkey = random.split(self.key)
                        seed_value = random.randint(subkey, (), 0, 1_000_000)
                        config = ModelConfig(seed=seed_value.item())
                        model = self.model_factory(params=self.params, config=config)

                        # Run the model for evaluation
                        results = model.run(steps=10)
                        
                        if metric in results and results[metric]:
                            value = results[metric][-1]
                            print(f"  Metric '{metric}': target={target:.4f}, current={value:.4f}")
                    
                    print(f"Current loss: {float(-reward):.6f}")
                
                # Early stopping if loss is small enough
                if -reward < 1e-6:
                    if verbose:
                        print("Early stopping: loss below threshold")
                    break
        
        if verbose:
            print("\nCalibration complete!")
            print(f"Final parameters: {', '.join([f'{k}={v:.4f}' for k, v in self.params.items()])}")
            print(f"Final loss: {self.loss_history[-1]:.6f}")
            
        return self.params
    
    def get_calibration_history(self) -> Dict[str, List[Any]]:
        """Get calibration history.
        
        Returns:
            Dictionary with 'loss' and 'params' histories
        """
        return {
            "loss": self.loss_history,
            "params": self.param_history
        }
    
    def plot_calibration(self, figsize: Tuple[int, int] = (12, 8)) -> Any:
        """Plot calibration results.
        
        Args:
            figsize: Figure size as (width, height)
            
        Returns:
            Matplotlib figure and axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is required for plotting. Install it with 'pip install matplotlib'")
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Plot loss history
        axes[0].plot(self.loss_history)
        axes[0].set_title("Loss over iterations")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True)
        
        # Plot parameter values
        param_names = list(self.params.keys())
        for param in param_names:
            values = [params[param] for params in self.param_history]
            axes[1].plot(values, label=param)
            
        axes[1].set_title("Parameter values over iterations")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Parameter value")
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        return fig, axes 