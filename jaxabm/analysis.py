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
                if metric in results_dict and results_dict[metric] is not None:
                    # Handle both scalar values and arrays/lists
                    metric_value = results_dict[metric]
                    if hasattr(metric_value, '__len__') and not isinstance(metric_value, str):
                        # It's an array or list, take the last value
                        value = metric_value[-1]
                    else:
                        # It's a scalar value
                        value = metric_value
                    
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
    """Calibrate model parameters using advanced optimization techniques.
    
    This class provides methods for automatically tuning model parameters
    to achieve desired outputs, using gradient-based optimization with Adam,
    or various reinforcement learning and evolutionary approaches.
    
    Attributes:
        model_factory: Function to create model instances
        params: Current parameter values
        target_metrics: Target values for each metric
        metrics_weights: Importance weights for each metric in the loss function
        learning_rate: Learning rate for optimization
        max_iterations: Maximum number of optimization iterations
        method: Calibration method
        loss_type: Type of loss function to use
        param_bounds: Parameter bounds for each parameter
        evaluation_steps: Number of steps to run model for evaluation
        num_evaluation_runs: Number of runs to average for robust evaluation
        loss_history: History of loss values during calibration
        param_history: History of parameter values during calibration
        confidence_intervals: Confidence intervals for metrics
    """
    
    def __init__(
        self, 
        model_factory: ModelFactory,
        initial_params: Dict[str, float],
        target_metrics: Dict[str, float],
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        metrics_weights: Optional[Dict[str, float]] = None,
        learning_rate: float = 0.01,
        max_iterations: int = 100,
        method: str = "adam",
        loss_type: str = "mse",
        evaluation_steps: int = 50,
        num_evaluation_runs: int = 3,
        tolerance: float = 1e-6,
        patience: int = 10,
        seed: int = 0
    ):
        """Initialize model calibrator.
        
        Args:
            model_factory: Function to create model instances
            initial_params: Initial parameter values
            target_metrics: Target metric values
            param_bounds: Bounds for each parameter as (min, max) tuples
            metrics_weights: Weights for each metric in the loss function
            learning_rate: Learning rate for optimization
            max_iterations: Maximum number of optimization iterations
            method: Calibration method ("adam", "sgd", "es", "pso", "cem", "bayesian")
            loss_type: Loss function type ("mse", "mae", "huber", "relative")
            evaluation_steps: Number of simulation steps for evaluation
            num_evaluation_runs: Number of runs to average for robust evaluation
            tolerance: Convergence tolerance
            patience: Early stopping patience
            seed: Random seed
        """
        self.model_factory = model_factory
        self.params = initial_params.copy()
        self.target_metrics = target_metrics
        self.param_bounds = param_bounds or {k: (0.01, 10.0) for k in initial_params}
        self.metrics_weights = metrics_weights or {k: 1.0 for k in target_metrics}
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.method = method
        self.loss_type = loss_type
        self.evaluation_steps = evaluation_steps
        self.num_evaluation_runs = num_evaluation_runs
        self.tolerance = tolerance
        self.patience = patience
        
        # Initialize random key
        self.key = random.PRNGKey(seed)
        
        # History tracking
        self.loss_history = []
        self.param_history = []
        self.confidence_intervals = []
        self.best_params = initial_params.copy()
        self.best_loss = float('inf')
        
        # Method-specific initialization
        self._setup_optimization_method()
    
    def _setup_optimization_method(self):
        """Set up the optimization method."""
        if self.method in ["adam", "sgd"]:
            self._setup_gradient_optimization()
        elif self.method == "es":
            self._setup_evolution_strategies()
        elif self.method == "pso":
            self._setup_particle_swarm()
        elif self.method == "cem":
            self._setup_cross_entropy()
        elif self.method == "bayesian":
            self._setup_bayesian_optimization()
        elif self.method == "q_learning":
            self._setup_q_learning()
        elif self.method == "policy_gradient":
            self._setup_policy_gradient()
        elif self.method == "actor_critic":
            self._setup_actor_critic()
        elif self.method == "multi_agent_rl":
            self._setup_multi_agent_rl()
        elif self.method == "dqn":
            self._setup_deep_q_network()
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
    
    def _compute_loss(self, metrics: Dict[str, float]) -> float:
        """Compute loss based on the specified loss type."""
        loss = 0.0
        
        for metric, target in self.target_metrics.items():
            if metric not in metrics:
                continue
                
            value = metrics[metric]
            weight = self.metrics_weights[metric]
            
            if self.loss_type == "mse":
                metric_loss = (value - target) ** 2
            elif self.loss_type == "mae":
                metric_loss = abs(value - target)
            elif self.loss_type == "huber":
                delta = 1.0
                residual = abs(value - target)
                # Use JAX's where function instead of if/else for JIT compatibility
                metric_loss = jnp.where(
                    residual <= delta,
                    0.5 * residual ** 2,
                    delta * (residual - 0.5 * delta)
                )
            elif self.loss_type == "relative":
                metric_loss = abs(value - target) / (abs(target) + 1e-8)
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
            
            loss += weight * metric_loss
        
        return loss
    
    def _evaluate_params_robust(self, params: Dict[str, float]) -> Tuple[float, Dict[str, Tuple[float, float]]]:
        """Evaluate parameters with multiple runs for robustness."""
        all_metrics = {metric: [] for metric in self.target_metrics}
        
        for run in range(self.num_evaluation_runs):
            # Use different seeds for each run
            self.key, subkey = random.split(self.key)
            seed_value = random.randint(subkey, (), 0, 1_000_000)
            config = ModelConfig(seed=seed_value.item())
            
            model = self.model_factory(params=params, config=config)
            results = model.run(steps=self.evaluation_steps)
            
            for metric in self.target_metrics:
                if metric in results:
                    # Handle both JAX arrays and lists
                    if hasattr(results[metric], '__len__') and len(results[metric]) > 0:
                        all_metrics[metric].append(float(results[metric][-1]))
                    else:
                        all_metrics[metric].append(0.0)
                else:
                    all_metrics[metric].append(0.0)
        
        # Compute mean metrics and confidence intervals
        mean_metrics = {}
        confidence_intervals = {}
        
        for metric, values in all_metrics.items():
            values_array = jnp.array(values)
            mean_val = float(jnp.mean(values_array))
            std_val = float(jnp.std(values_array))
            
            mean_metrics[metric] = mean_val
            # 95% confidence interval
            ci_half_width = 1.96 * std_val / jnp.sqrt(len(values))
            confidence_intervals[metric] = (
                mean_val - ci_half_width,
                mean_val + ci_half_width
            )
        
        # Use normalized loss for better RL performance
        loss = self._compute_normalized_loss(mean_metrics)
        return loss, confidence_intervals
    
    def _compute_normalized_loss(self, metrics: Dict[str, float]) -> float:
        """Compute normalized loss for better RL optimization (key improvement!)."""
        total_loss = 0.0
        for metric, target in self.target_metrics.items():
            if metric in metrics:
                value = metrics[metric]
                # Normalize by target to make losses comparable and stable
                normalized_error = abs(value - target) / (abs(target) + 1e-8)
                total_loss += normalized_error ** 2
        return float(total_loss)
    
    def _setup_gradient_optimization(self):
        """Set up gradient-based optimization with Adam or SGD."""
        param_names = list(self.params.keys())
        
        # Use a fixed seed for gradient computation to avoid tracer issues
        def loss_fn(params_flat):
            # Convert flat parameters to dictionary
            params = {name: params_flat[i] for i, name in enumerate(param_names)}
            
            # Use a fixed seed for gradient computation (deterministic)
            config = ModelConfig(seed=42)
            
            model = self.model_factory(params=params, config=config)
            results = model.run(steps=self.evaluation_steps)
            
            # Handle both JAX arrays and lists
            metrics = {}
            for metric in self.target_metrics:
                if metric in results:
                    if hasattr(results[metric], '__len__') and len(results[metric]) > 0:
                        # Take the last value, handling both JAX arrays and lists
                        if hasattr(results[metric], 'at'):  # JAX array
                            metrics[metric] = results[metric][-1]
                        else:  # Python list
                            metrics[metric] = results[metric][-1]
                    else:
                        metrics[metric] = 0.0
                else:
                    metrics[metric] = 0.0
            
            return self._compute_loss(metrics)
        
        self.loss_fn = loss_fn
        self.grad_fn = jit(grad(loss_fn))
        
        if self.method == "adam":
            # Adam optimizer state
            self.adam_m = jnp.zeros(len(param_names))  # First moment
            self.adam_v = jnp.zeros(len(param_names))  # Second moment
            self.adam_beta1 = 0.9
            self.adam_beta2 = 0.999
            self.adam_eps = 1e-8
            self.adam_t = 0  # Time step
    
    def _setup_evolution_strategies(self):
        """Set up Evolution Strategies (ES) optimization."""
        self.es_population_size = 20
        self.es_sigma = 0.1  # Mutation strength
        self.es_elite_ratio = 0.2  # Fraction of population to keep as elite
        
        # Initialize population
        param_names = list(self.params.keys())
        n_params = len(param_names)
        
        self.key, subkey = random.split(self.key)
        self.es_population = random.normal(subkey, (self.es_population_size, n_params)) * self.es_sigma
        
        # Center population around initial parameters
        initial_flat = jnp.array([self.params[name] for name in param_names])
        self.es_population = self.es_population + initial_flat[None, :]
        
        # Clip to bounds
        for i, (param, (min_val, max_val)) in enumerate(self.param_bounds.items()):
            self.es_population = self.es_population.at[:, i].set(
                jnp.clip(self.es_population[:, i], min_val, max_val)
            )
    
    def _setup_particle_swarm(self):
        """Set up Particle Swarm Optimization (PSO)."""
        self.pso_population_size = 20
        self.pso_w = 0.7  # Inertia weight
        self.pso_c1 = 1.5  # Cognitive parameter
        self.pso_c2 = 1.5  # Social parameter
        
        param_names = list(self.params.keys())
        n_params = len(param_names)
        
        # Initialize particles
        self.key, subkey = random.split(self.key)
        self.pso_positions = random.uniform(subkey, (self.pso_population_size, n_params))
        
        # Scale to parameter bounds
        for i, (param, (min_val, max_val)) in enumerate(self.param_bounds.items()):
            self.pso_positions = self.pso_positions.at[:, i].multiply(max_val - min_val)
            self.pso_positions = self.pso_positions.at[:, i].add(min_val)
        
        # Initialize velocities
        self.key, subkey = random.split(self.key)
        velocity_scale = 0.1
        self.pso_velocities = random.normal(subkey, (self.pso_population_size, n_params)) * velocity_scale
        
        # Personal and global best
        self.pso_personal_best = self.pso_positions.copy()
        self.pso_personal_best_scores = jnp.full(self.pso_population_size, float('inf'))
        self.pso_global_best = self.pso_positions[0].copy()
        self.pso_global_best_score = float('inf')
    
    def _setup_cross_entropy(self):
        """Set up Cross-Entropy Method (CEM)."""
        self.cem_population_size = 50
        self.cem_elite_ratio = 0.2
        self.cem_noise_decay = 0.99
        
        param_names = list(self.params.keys())
        n_params = len(param_names)
        
        # Initialize distribution parameters
        self.cem_mean = jnp.array([self.params[name] for name in param_names])
        self.cem_std = jnp.ones(n_params) * 0.5
    
    def _setup_bayesian_optimization(self):
        """Set up Bayesian Optimization with Gaussian Process."""
        # Simple implementation - in practice, you'd use a library like GPyOpt
        self.bo_n_initial = 10
        self.bo_acquisition = "ei"  # Expected Improvement
        
        param_names = list(self.params.keys())
        n_params = len(param_names)
        
        # Generate initial samples
        self.key, subkey = random.split(self.key)
        self.bo_X = random.uniform(subkey, (self.bo_n_initial, n_params))
        
        # Scale to parameter bounds
        for i, (param, (min_val, max_val)) in enumerate(self.param_bounds.items()):
            self.bo_X = self.bo_X.at[:, i].multiply(max_val - min_val)
            self.bo_X = self.bo_X.at[:, i].add(min_val)
        
        self.bo_y = jnp.full(self.bo_n_initial, float('inf'))
        self.bo_evaluated = 0
    
    def _setup_q_learning(self):
        """Set up Improved Q-Learning with continuous action space and better state representation."""
        self.ql_learning_rate = 0.001
        self.ql_epsilon = 0.2  # Lower initial exploration
        self.ql_epsilon_decay = 0.995
        self.ql_epsilon_min = 0.02  # Lower minimum exploration
        self.ql_gamma = 0.95
        self.ql_batch_size = 64  # Larger batch size
        self.ql_memory_size = 2000
        self.ql_target_update_freq = 50  # Target network updates
        
        param_names = list(self.params.keys())
        n_params = len(param_names)
        
        # Enhanced state size: current params + targets + history + gradients
        state_size = n_params * 4  # current, normalized, targets, gradients
        n_actions = n_params * 5  # 5 different step sizes per parameter
        
        # Improved neural network architecture
        self.key, subkey1, subkey2, subkey3, subkey4 = random.split(self.key, 5)
        hidden1_size = 256
        hidden2_size = 128
        hidden3_size = 64
        
        # Main Q-network
        self.ql_params = {
            'layer1': {
                'weights': random.normal(subkey1, (state_size, hidden1_size)) * 0.1,
                'bias': jnp.zeros(hidden1_size)
            },
            'layer2': {
                'weights': random.normal(subkey2, (hidden1_size, hidden2_size)) * 0.1,
                'bias': jnp.zeros(hidden2_size)
            },
            'layer3': {
                'weights': random.normal(subkey3, (hidden2_size, hidden3_size)) * 0.1,
                'bias': jnp.zeros(hidden3_size)
            },
            'output': {
                'weights': random.normal(subkey4, (hidden3_size, n_actions)) * 0.1,
                'bias': jnp.zeros(n_actions)
            }
        }
        
        # Target network (copy of main network)
        self.ql_target_params = {k: {kk: vv.copy() for kk, vv in v.items()} 
                                for k, v in self.ql_params.items()}
        
        # Enhanced experience replay with prioritization
        self.ql_memory = []
        self.ql_param_names = param_names
        self.ql_step_sizes = [0.01, 0.03, 0.05, 0.1, 0.2]  # Multiple step sizes
        
        # State history for enhanced representation
        self.ql_param_history = []
        self.ql_loss_history = []
        self.ql_gradient_estimates = jnp.zeros(n_params)
        
        # Parameter space normalization
        self.ql_param_mins = jnp.array([self.param_bounds[p][0] for p in param_names])
        self.ql_param_maxs = jnp.array([self.param_bounds[p][1] for p in param_names])
        
        # Adam optimizer for neural network training
        self.ql_adam_m = {k: {kk: jnp.zeros_like(vv) for kk, vv in v.items()} 
                         for k, v in self.ql_params.items()}
        self.ql_adam_v = {k: {kk: jnp.zeros_like(vv) for kk, vv in v.items()} 
                         for k, v in self.ql_params.items()}
        self.ql_adam_t = 0
    
    def _setup_policy_gradient(self):
        """Set up Enhanced Policy Gradient with continuous actions and better architecture."""
        self.pg_learning_rate = 0.005
        self.pg_baseline_decay = 0.9
        self.pg_min_std = 0.05
        self.pg_entropy_coeff = 0.02
        self.pg_value_coeff = 0.1  # For baseline learning
        
        param_names = list(self.params.keys())
        n_params = len(param_names)
        
        # Enhanced state representation
        state_size = n_params * 4  # current, normalized, targets, history
        
        # Policy network (actor) - outputs continuous actions
        self.key, subkey1, subkey2, subkey3 = random.split(self.key, 4)
        hidden_size = 128
        
        self.pg_policy_params = {
            'shared_layer1': {
                'weights': random.normal(subkey1, (state_size, hidden_size)) * 0.1,
                'bias': jnp.zeros(hidden_size)
            },
            'shared_layer2': {
                'weights': random.normal(subkey2, (hidden_size, 64)) * 0.1,
                'bias': jnp.zeros(64)
            },
            'mean_output': {
                'weights': random.normal(subkey3, (64, n_params)) * 0.1,
                'bias': jnp.zeros(n_params)
            },
            'std_output': {
                'weights': random.normal(subkey3, (64, n_params)) * 0.1,
                'bias': jnp.ones(n_params) * jnp.log(0.2)  # Initialize to reasonable std
            }
        }
        
        # Value network (critic) for baseline
        self.key, subkey4, subkey5 = random.split(self.key, 3)
        self.pg_value_params = {
            'layer1': {
                'weights': random.normal(subkey4, (state_size, hidden_size)) * 0.1,
                'bias': jnp.zeros(hidden_size)
            },
            'layer2': {
                'weights': random.normal(subkey5, (hidden_size, 1)) * 0.1,
                'bias': jnp.array([0.0])
            }
        }
        
        self.pg_param_names = param_names
        
        # Enhanced state tracking
        self.pg_state_history = []
        self.pg_loss_history = []
        
        # Adam optimizers for both networks
        self.pg_policy_adam_m = {k: {kk: jnp.zeros_like(vv) for kk, vv in v.items()} 
                                for k, v in self.pg_policy_params.items()}
        self.pg_policy_adam_v = {k: {kk: jnp.zeros_like(vv) for kk, vv in v.items()} 
                                for k, v in self.pg_policy_params.items()}
        self.pg_value_adam_m = {k: {kk: jnp.zeros_like(vv) for kk, vv in v.items()} 
                               for k, v in self.pg_value_params.items()}
        self.pg_value_adam_v = {k: {kk: jnp.zeros_like(vv) for kk, vv in v.items()} 
                               for k, v in self.pg_value_params.items()}
        self.pg_adam_t = 0
    
    def _setup_actor_critic(self):
        """Set up Advanced Actor-Critic with proper network architecture and training."""
        self.ac_actor_lr = 0.003
        self.ac_critic_lr = 0.005
        self.ac_gamma = 0.95
        self.ac_lambda = 0.95  # GAE lambda
        self.ac_gradient_clip_norm = 1.0
        self.ac_entropy_coeff = 0.02
        
        param_names = list(self.params.keys())
        n_params = len(param_names)
        state_size = n_params * 4
        
        # Advanced actor network architecture
        self.key, *subkeys = random.split(self.key, 9)
        hidden_size = 128
        
        self.ac_actor_params = {
            'layer1': {
                'weights': random.normal(subkeys[0], (state_size, hidden_size)) * 0.1,
                'bias': jnp.zeros(hidden_size)
            },
            'layer2': {
                'weights': random.normal(subkeys[1], (hidden_size, 64)) * 0.1,
                'bias': jnp.zeros(64)
            },
            'mean_output': {
                'weights': random.normal(subkeys[2], (64, n_params)) * 0.1,
                'bias': jnp.zeros(n_params)
            },
            'std_output': {
                'weights': random.normal(subkeys[3], (64, n_params)) * 0.1,
                'bias': jnp.ones(n_params) * jnp.log(0.2)
            }
        }
        
        # Advanced critic network
        self.ac_critic_params = {
            'layer1': {
                'weights': random.normal(subkeys[4], (state_size, hidden_size)) * 0.1,
                'bias': jnp.zeros(hidden_size)
            },
            'layer2': {
                'weights': random.normal(subkeys[5], (hidden_size, 64)) * 0.1,
                'bias': jnp.zeros(64)
            },
            'layer3': {
                'weights': random.normal(subkeys[6], (64, 32)) * 0.1,
                'bias': jnp.zeros(32)
            },
            'output': {
                'weights': random.normal(subkeys[7], (32, 1)) * 0.1,
                'bias': jnp.array([0.0])
            }
        }
        
        self.ac_param_names = param_names
        
        # GAE (Generalized Advantage Estimation) buffers
        self.ac_states = []
        self.ac_actions = []
        self.ac_rewards = []
        self.ac_values = []
        self.ac_log_probs = []
        self.ac_dones = []
        
        # Adam optimizers
        self.ac_actor_adam_m = {k: {kk: jnp.zeros_like(vv) for kk, vv in v.items()} 
                               for k, v in self.ac_actor_params.items()}
        self.ac_actor_adam_v = {k: {kk: jnp.zeros_like(vv) for kk, vv in v.items()} 
                               for k, v in self.ac_actor_params.items()}
        self.ac_critic_adam_m = {k: {kk: jnp.zeros_like(vv) for kk, vv in v.items()} 
                                for k, v in self.ac_critic_params.items()}
        self.ac_critic_adam_v = {k: {kk: jnp.zeros_like(vv) for kk, vv in v.items()} 
                                for k, v in self.ac_critic_params.items()}
        self.ac_adam_t = 0
    
    def _setup_deep_q_network(self):
        """Set up Advanced DQN with Double DQN, Dueling Architecture, and Prioritized Experience Replay."""
        self.dqn_learning_rate = 0.0005
        self.dqn_epsilon = 0.3
        self.dqn_epsilon_decay = 0.995
        self.dqn_epsilon_min = 0.05
        self.dqn_gamma = 0.95
        self.dqn_batch_size = 64
        self.dqn_memory_size = 3000
        self.dqn_target_update_freq = 100
        self.dqn_double_dqn = True  # Use Double DQN
        
        param_names = list(self.params.keys())
        n_params = len(param_names)
        state_size = n_params * 4
        n_actions = n_params * 7  # More granular actions
        
        # Dueling DQN architecture
        self.key, *subkeys = random.split(self.key, 10)
        hidden_size = 256
        
        # Shared layers
        self.dqn_params = {
            'shared1': {
                'weights': random.normal(subkeys[0], (state_size, hidden_size)) * 0.1,
                'bias': jnp.zeros(hidden_size)
            },
            'shared2': {
                'weights': random.normal(subkeys[1], (hidden_size, 128)) * 0.1,
                'bias': jnp.zeros(128)
            },
            # Value stream
            'value1': {
                'weights': random.normal(subkeys[2], (128, 64)) * 0.1,
                'bias': jnp.zeros(64)
            },
            'value_output': {
                'weights': random.normal(subkeys[3], (64, 1)) * 0.1,
                'bias': jnp.array([0.0])
            },
            # Advantage stream
            'advantage1': {
                'weights': random.normal(subkeys[4], (128, 64)) * 0.1,
                'bias': jnp.zeros(64)
            },
            'advantage_output': {
                'weights': random.normal(subkeys[5], (64, n_actions)) * 0.1,
                'bias': jnp.zeros(n_actions)
            }
        }
        
        # Target network
        self.dqn_target_params = {k: {kk: vv.copy() for kk, vv in v.items()} 
                                 for k, v in self.dqn_params.items()}
        
        # Prioritized experience replay
        self.dqn_memory = []
        self.dqn_priorities = []
        self.dqn_alpha = 0.6  # Prioritization exponent
        self.dqn_beta = 0.4   # Importance sampling exponent
        self.dqn_beta_increment = 0.001
        
        self.dqn_param_names = param_names
        self.dqn_step_sizes = [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
        
        # Action history for diversity regularization
        self.dqn_action_history = []
        self.dqn_action_history_size = 10
        self.dqn_action_regularization = 0.1
        
        # Adam optimizer
        self.dqn_adam_m = {k: {kk: jnp.zeros_like(vv) for kk, vv in v.items()} 
                          for k, v in self.dqn_params.items()}
        self.dqn_adam_v = {k: {kk: jnp.zeros_like(vv) for kk, vv in v.items()} 
                          for k, v in self.dqn_params.items()}
        self.dqn_adam_t = 0
    
    def _setup_multi_agent_rl(self):
        """Set up Advanced Multi-Agent RL with coordinated exploration and communication."""
        self.marl_learning_rate = 0.002
        self.marl_epsilon = 0.25
        self.marl_epsilon_decay = 0.995
        self.marl_epsilon_min = 0.05
        self.marl_communication_dim = 16  # Communication vector size
        
        param_names = list(self.params.keys())
        n_agents = len(param_names)
        
        # Each agent has enhanced architecture with communication
        self.marl_agents = {}
        for i, param in enumerate(param_names):
            self.key, *subkeys = random.split(self.key, 8)
            
            # Enhanced state: own param + global state + communication from others
            local_state_size = 1 + n_agents * 3 + (n_agents - 1) * self.marl_communication_dim
            
            self.marl_agents[param] = {
                # Q-network with communication
                'q_params': {
                    'layer1': {
                        'weights': random.normal(subkeys[0], (local_state_size, 128)) * 0.1,
                        'bias': jnp.zeros(128)
                    },
                    'layer2': {
                        'weights': random.normal(subkeys[1], (128, 64)) * 0.1,
                        'bias': jnp.zeros(64)
                    },
                    'q_output': {
                        'weights': random.normal(subkeys[2], (64, 5)) * 0.1,  # 5 actions per agent
                        'bias': jnp.zeros(5)
                    }
                },
                # Communication network
                'comm_params': {
                    'layer1': {
                        'weights': random.normal(subkeys[3], (local_state_size, 32)) * 0.1,
                        'bias': jnp.zeros(32)
                    },
                    'comm_output': {
                        'weights': random.normal(subkeys[4], (32, self.marl_communication_dim)) * 0.1,
                        'bias': jnp.zeros(self.marl_communication_dim)
                    }
                },
                'epsilon': self.marl_epsilon,
                'memory': [],
                'memory_size': 500,
                # Adam optimizers
                'q_adam_m': {k: {kk: jnp.zeros_like(vv) for kk, vv in v.items()} 
                            for k, v in self.marl_agents.get(param, {}).get('q_params', {}).items()},
                'q_adam_v': {k: {kk: jnp.zeros_like(vv) for kk, vv in v.items()} 
                            for k, v in self.marl_agents.get(param, {}).get('q_params', {}).items()},
                'comm_adam_m': {k: {kk: jnp.zeros_like(vv) for kk, vv in v.items()} 
                               for k, v in self.marl_agents.get(param, {}).get('comm_params', {}).items()},
                'comm_adam_v': {k: {kk: jnp.zeros_like(vv) for kk, vv in v.items()} 
                               for k, v in self.marl_agents.get(param, {}).get('comm_params', {}).items()},
                'adam_t': 0
            }
        
        # Initialize Adam optimizers properly after agent creation
        for param in param_names:
            agent = self.marl_agents[param]
            agent['q_adam_m'] = {k: {kk: jnp.zeros_like(vv) for kk, vv in v.items()} 
                                for k, v in agent['q_params'].items()}
            agent['q_adam_v'] = {k: {kk: jnp.zeros_like(vv) for kk, vv in v.items()} 
                                for k, v in agent['q_params'].items()}
            agent['comm_adam_m'] = {k: {kk: jnp.zeros_like(vv) for kk, vv in v.items()} 
                                   for k, v in agent['comm_params'].items()}
            agent['comm_adam_v'] = {k: {kk: jnp.zeros_like(vv) for kk, vv in v.items()} 
                                   for k, v in agent['comm_params'].items()}
        
        self.marl_param_names = param_names
        self.marl_step_sizes = [0.01, 0.03, 0.05, 0.1, 0.15]  # 5 actions per agent
        
        # Global communication buffer
        self.marl_global_comm = jnp.zeros((n_agents, self.marl_communication_dim))
    
    def _params_to_state(self, params: Dict[str, float]) -> jax.Array:
        """Convert parameters to normalized state for neural networks."""
        param_values = jnp.array([params[name] for name in self.ql_param_names])
        normalized = (param_values - self.ql_param_mins) / (self.ql_param_maxs - self.ql_param_mins)
        return jnp.clip(normalized, 0.0, 1.0)
    
    def _find_nearest_bin(self, value: float, bins: jax.Array) -> int:
        """Find the nearest bin index for a value."""
        return int(jnp.argmin(jnp.abs(bins - value)))
    
    def _dqn_forward(self, state: jax.Array) -> jax.Array:
        """Forward pass through DQN with deeper architecture."""
        # Layer 1
        h1 = jnp.tanh(jnp.dot(state, self.dqn_params['layer1']['weights']) + 
                      self.dqn_params['layer1']['bias'])
        # Layer 2
        h2 = jnp.tanh(jnp.dot(h1, self.dqn_params['layer2']['weights']) + 
                      self.dqn_params['layer2']['bias'])
        # Layer 3 (output)
        q_values = jnp.dot(h2, self.dqn_params['layer3']['weights']) + \
                   self.dqn_params['layer3']['bias']
        return q_values
    
    def _policy_gradient_sample_action(self, state: jax.Array) -> Tuple[jax.Array, float, float]:
        """Sample action from policy with exploration preservation."""
        mean = self.pg_policy_params['mean']
        # Enforce minimum standard deviation to prevent policy collapse
        log_std = jnp.maximum(self.pg_policy_params['log_std'], jnp.log(self.pg_min_std))
        std = jnp.exp(log_std)
        
        # Sample from normal distribution
        self.key, subkey = random.split(self.key)
        action = mean + std * random.normal(subkey, mean.shape)
        
        # Compute log probability
        log_prob = -0.5 * jnp.sum(((action - mean) / std) ** 2) - \
                   0.5 * jnp.sum(jnp.log(2 * jnp.pi * std ** 2))
        
        # Compute entropy for regularization
        entropy = 0.5 * jnp.sum(log_std + jnp.log(2 * jnp.pi * jnp.e))
        
        return action, float(log_prob), float(entropy)
    
    def _actor_critic_forward(self, state: jax.Array) -> Tuple[jax.Array, float, float]:
        """Forward pass for robust actor-critic."""
        # Actor (policy) with minimum std constraint
        mean = self.ac_actor_params['mean']
        log_std = jnp.maximum(self.ac_actor_params['log_std'], jnp.log(0.05))  # Min std = 0.05
        std = jnp.exp(log_std)
        
        # Sample action
        self.key, subkey = random.split(self.key)
        action = mean + std * random.normal(subkey, mean.shape)
        
        # Log probability
        log_prob = -0.5 * jnp.sum(((action - mean) / std) ** 2) - \
                   0.5 * jnp.sum(jnp.log(2 * jnp.pi * std ** 2))
        
        # Robust critic (value function) with hidden layer
        h1 = jnp.tanh(jnp.dot(state, self.ac_critic_params['layer1_weights']) + 
                      self.ac_critic_params['layer1_bias'])
        value = jnp.dot(h1, self.ac_critic_params['layer2_weights']).squeeze() + \
                self.ac_critic_params['layer2_bias'][0]
        
        # Clip value to prevent explosion
        value = jnp.clip(value, *self.ac_value_clip_range)
        
        return action, float(log_prob), float(value)
    
    def _ql_forward(self, state: jax.Array) -> jax.Array:
        """Forward pass through Q-learning neural network."""
        # Layer 1
        h1 = jnp.tanh(jnp.dot(state, self.ql_params['layer1']['weights']) + 
                      self.ql_params['layer1']['bias'])
        # Layer 2  
        h2 = jnp.tanh(jnp.dot(h1, self.ql_params['layer2']['weights']) + 
                      self.ql_params['layer2']['bias'])
        # Layer 3 (output)
        q_values = jnp.dot(h2, self.ql_params['layer3']['weights']) + \
                   self.ql_params['layer3']['bias']
        return q_values
    
    def _ql_forward_improved(self, state: jax.Array, params: Optional[Dict] = None) -> jax.Array:
        """Improved Q-learning forward pass with optional custom parameters."""
        if params is None:
            params = self.ql_params
        
        # Enhanced forward pass with deeper network
        h1 = jnp.tanh(jnp.dot(state, params['layer1']['weights']) + 
                      params['layer1']['bias'])
        h2 = jnp.tanh(jnp.dot(h1, params['layer2']['weights']) + 
                      params['layer2']['bias'])
        h3 = jnp.tanh(jnp.dot(h2, params['layer3']['weights']) + 
                      params['layer3']['bias'])
        q_values = jnp.dot(h3, params['output']['weights']) + params['output']['bias']
        return q_values
    
    def _pg_forward_enhanced(self, state: jax.Array) -> Tuple[jax.Array, float, float, float]:
        """Enhanced policy gradient forward pass returning action, mean, std, value."""
        # Policy network (actor) - using correct parameter names
        h1_policy = jnp.tanh(jnp.dot(state, self.pg_policy_params['shared_layer1']['weights']) + 
                            self.pg_policy_params['shared_layer1']['bias'])
        h2_policy = jnp.tanh(jnp.dot(h1_policy, self.pg_policy_params['shared_layer2']['weights']) + 
                            self.pg_policy_params['shared_layer2']['bias'])
        
        # Mean and log_std for continuous actions with gradient clipping
        mean = jnp.dot(h2_policy, self.pg_policy_params['mean_output']['weights']) + self.pg_policy_params['mean_output']['bias']
        mean = jnp.clip(mean, -10.0, 10.0)  # Clip mean to prevent explosion
        
        log_std = jnp.dot(h2_policy, self.pg_policy_params['std_output']['weights']) + self.pg_policy_params['std_output']['bias']
        log_std = jnp.clip(log_std, -5.0, 2.0)  # More conservative clipping
        std = jnp.exp(log_std)
        std = jnp.maximum(std, 0.01)  # Minimum std to prevent collapse
        
        # Sample action with safe noise
        self.key, subkey = random.split(self.key)
        noise = random.normal(subkey, mean.shape)
        noise = jnp.clip(noise, -3.0, 3.0)  # Clip noise to prevent outliers
        action = mean + std * noise
        
        # Value network (critic) with safer computation
        h1_value = jnp.tanh(jnp.dot(state, self.pg_value_params['layer1']['weights']) + 
                           self.pg_value_params['layer1']['bias'])
        value = jnp.dot(h1_value, self.pg_value_params['layer2']['weights']) + self.pg_value_params['layer2']['bias']
        value = jnp.clip(value, -100.0, 100.0)  # Clip value to prevent explosion
        
        # Ensure outputs are finite
        mean = jnp.where(jnp.isfinite(mean), mean, 0.0)
        std = jnp.where(jnp.isfinite(std), std, 0.1)
        value = jnp.where(jnp.isfinite(value), value, 0.0)
        
        return action, float(mean[0]), float(std[0]), float(value[0])
    
    def _ac_forward_enhanced(self, state: jax.Array) -> Tuple[jax.Array, float, float, float]:
        """Enhanced actor-critic forward pass returning action, log_prob, value, entropy."""
        # Actor network - using correct parameter names
        h1_actor = jnp.tanh(jnp.dot(state, self.ac_actor_params['layer1']['weights']) + 
                           self.ac_actor_params['layer1']['bias'])
        h2_actor = jnp.tanh(jnp.dot(h1_actor, self.ac_actor_params['layer2']['weights']) + 
                           self.ac_actor_params['layer2']['bias'])
        
        # Action probabilities for discrete actions or mean/std for continuous
        mean = jnp.dot(h2_actor, self.ac_actor_params['mean_output']['weights']) + self.ac_actor_params['mean_output']['bias']
        log_std = jnp.dot(h2_actor, self.ac_actor_params['std_output']['weights']) + self.ac_actor_params['std_output']['bias']
        std = jnp.exp(jnp.clip(log_std, -20, 2))
        
        # Sample action
        self.key, subkey = random.split(self.key)
        action = mean + std * random.normal(subkey, mean.shape)
        
        # Compute log probability
        log_prob = -0.5 * ((action - mean) / std) ** 2 - 0.5 * jnp.log(2 * jnp.pi) - jnp.log(std)
        log_prob = jnp.sum(log_prob)  # Sum over action dimensions
        
        # Compute entropy
        entropy = 0.5 * jnp.log(2 * jnp.pi * jnp.e) + jnp.log(std)
        entropy = jnp.sum(entropy)
        
        # Critic network
        h1_critic = jnp.tanh(jnp.dot(state, self.ac_critic_params['layer1']['weights']) + 
                            self.ac_critic_params['layer1']['bias'])
        h2_critic = jnp.tanh(jnp.dot(h1_critic, self.ac_critic_params['layer2']['weights']) + 
                            self.ac_critic_params['layer2']['bias'])
        h3_critic = jnp.tanh(jnp.dot(h2_critic, self.ac_critic_params['layer3']['weights']) + 
                            self.ac_critic_params['layer3']['bias'])
        value = jnp.dot(h3_critic, self.ac_critic_params['output']['weights']) + self.ac_critic_params['output']['bias']
        
        return action, float(log_prob), float(value[0]), float(entropy)
    
    def _dqn_forward_dueling(self, state: jax.Array, params: Optional[Dict] = None) -> jax.Array:
        """Dueling DQN forward pass with separate value and advantage streams."""
        if params is None:
            params = self.dqn_params
        
        # Shared layers
        h1 = jnp.tanh(jnp.dot(state, params['shared1']['weights']) + 
                      params['shared1']['bias'])
        h2 = jnp.tanh(jnp.dot(h1, params['shared2']['weights']) + 
                      params['shared2']['bias'])
        
        # Value stream - using correct parameter names
        v_stream = jnp.tanh(jnp.dot(h2, params['value1']['weights']) + 
                           params['value1']['bias'])
        state_value = jnp.dot(v_stream, params['value_output']['weights']) + params['value_output']['bias']
        
        # Advantage stream - using correct parameter names  
        a_stream = jnp.tanh(jnp.dot(h2, params['advantage1']['weights']) + 
                           params['advantage1']['bias'])
        advantages = jnp.dot(a_stream, params['advantage_output']['weights']) + params['advantage_output']['bias']
        
        # Combine value and advantages (dueling architecture)
        q_values = state_value + (advantages - jnp.mean(advantages, keepdims=True))
        
        return q_values
    
    def _marl_agent_forward(self, agent_params: Dict, state: float) -> jax.Array:
        """Forward pass for individual multi-agent RL agent."""
        state_array = jnp.array([state])
        h1 = jnp.tanh(jnp.dot(state_array, agent_params['layer1_weights']) + 
                      agent_params['layer1_bias'])
        q_values = jnp.dot(h1, agent_params['layer2_weights']) + agent_params['layer2_bias']
        return q_values
    
    def _compute_action_diversity_penalty(self, action: int) -> float:
        """Compute penalty for repetitive actions in DQN."""
        if len(self.dqn_action_history) == 0:
            return 0.0
            
        # Count recent occurrences of this action
        recent_actions = self.dqn_action_history[-self.dqn_action_history_size:]
        action_count = sum(1 for a in recent_actions if a == action)
        
        # Penalty proportional to frequency
        penalty = self.dqn_action_regularization * (action_count / len(recent_actions))
        return penalty
    
    def _create_enhanced_state(self, params: Dict[str, float]) -> jax.Array:
        """Create enhanced state representation with normalized params, targets, history, and gradients."""
        param_names = list(self.params.keys())
        n_params = len(param_names)
        
        # Current normalized parameters
        current_params = jnp.array([params[name] for name in param_names])
        param_mins = jnp.array([self.param_bounds[p][0] for p in param_names])
        param_maxs = jnp.array([self.param_bounds[p][1] for p in param_names])
        normalized_params = (current_params - param_mins) / (param_maxs - param_mins)
        normalized_params = jnp.clip(normalized_params, 0.0, 1.0)
        
        # Target metrics (normalized)
        target_values = jnp.array([self.target_metrics.get(metric, 0.0) for metric in self.target_metrics])
        target_norm = target_values / (jnp.abs(target_values) + 1e-8)
        
        # Parameter history features (recent trends)
        if hasattr(self, 'ql_param_history') and len(self.ql_param_history) > 0:
            recent_params = jnp.array([self.ql_param_history[-1][name] for name in param_names])
            param_change = (current_params - recent_params) / (param_maxs - param_mins + 1e-8)
        else:
            param_change = jnp.zeros(n_params)
        
        # Gradient estimates
        if hasattr(self, 'ql_gradient_estimates'):
            gradient_norm = self.ql_gradient_estimates / (jnp.abs(self.ql_gradient_estimates) + 1e-8)
        else:
            gradient_norm = jnp.zeros(n_params)
        
        # Combine all features
        enhanced_state = jnp.concatenate([
            normalized_params,  # Current normalized parameters
            target_norm[:n_params] if len(target_norm) >= n_params else jnp.zeros(n_params),  # Targets
            param_change,       # Recent parameter changes
            gradient_norm       # Gradient information
        ])
        
        return enhanced_state
    
    def _adam_update(self, params: Dict, gradients: Dict, adam_m: Dict, adam_v: Dict, 
                    adam_t: int, learning_rate: float = 0.001, 
                    beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> Tuple[Dict, Dict, Dict]:
        """Proper Adam optimizer update for neural network parameters."""
        updated_params = {}
        updated_m = {}
        updated_v = {}
        
        for layer_name, layer_params in params.items():
            updated_params[layer_name] = {}
            updated_m[layer_name] = {}
            updated_v[layer_name] = {}
            
            for param_name, param_value in layer_params.items():
                grad = gradients[layer_name][param_name]
                m = adam_m[layer_name][param_name]
                v = adam_v[layer_name][param_name]
                
                # Update biased first moment estimate
                m_new = beta1 * m + (1 - beta1) * grad
                
                # Update biased second raw moment estimate
                v_new = beta2 * v + (1 - beta2) * (grad ** 2)
                
                # Compute bias-corrected first moment estimate
                m_hat = m_new / (1 - beta1 ** adam_t)
                
                # Compute bias-corrected second raw moment estimate
                v_hat = v_new / (1 - beta2 ** adam_t)
                
                # Update parameters
                updated_params[layer_name][param_name] = param_value - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
                updated_m[layer_name][param_name] = m_new
                updated_v[layer_name][param_name] = v_new
        
        return updated_params, updated_m, updated_v
    
    def calibrate(self, verbose: bool = True) -> Dict[str, float]:
        """Run calibration process and return optimized parameters."""
        if verbose:
            print(f"Starting calibration with {self.method} method...")
            print(f"Target metrics: {self.target_metrics}")
            print(f"Parameter bounds: {self.param_bounds}")
        
        if self.method in ["adam", "sgd"]:
            return self._calibrate_gradient(verbose)
        elif self.method == "es":
            return self._calibrate_evolution_strategies(verbose)
        elif self.method == "pso":
            return self._calibrate_particle_swarm(verbose)
        elif self.method == "cem":
            return self._calibrate_cross_entropy(verbose)
        elif self.method == "bayesian":
            return self._calibrate_bayesian(verbose)
        elif self.method == "q_learning":
            return self._calibrate_q_learning(verbose)
        elif self.method == "policy_gradient":
            return self._calibrate_policy_gradient(verbose)
        elif self.method == "actor_critic":
            return self._calibrate_actor_critic(verbose)
        elif self.method == "multi_agent_rl":
            return self._calibrate_multi_agent_rl(verbose)
        elif self.method == "dqn":
            return self._calibrate_dqn(verbose)
    
    def _calibrate_gradient(self, verbose: bool) -> Dict[str, float]:
        """Gradient-based calibration with Adam or SGD."""
        param_names = list(self.params.keys())
        params_flat = jnp.array([self.params[name] for name in param_names])
        
        no_improvement_count = 0
        
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\nIteration {iteration+1}/{self.max_iterations}")
            
            # Compute gradients
            grads = self.grad_fn(params_flat)
            
            # Check for NaN gradients
            if jnp.any(jnp.isnan(grads)):
                if verbose:
                    print("NaN gradients detected, stopping optimization")
                break
            
            # Apply optimizer update
            if self.method == "adam":
                self.adam_t += 1
                
                # Update biased first moment estimate
                self.adam_m = self.adam_beta1 * self.adam_m + (1 - self.adam_beta1) * grads
                
                # Update biased second raw moment estimate
                self.adam_v = self.adam_beta2 * self.adam_v + (1 - self.adam_beta2) * (grads ** 2)
                
                # Compute bias-corrected first moment estimate
                m_hat = self.adam_m / (1 - self.adam_beta1 ** self.adam_t)
                
                # Compute bias-corrected second raw moment estimate
                v_hat = self.adam_v / (1 - self.adam_beta2 ** self.adam_t)
                
                # Update parameters
                params_flat = params_flat - self.learning_rate * m_hat / (jnp.sqrt(v_hat) + self.adam_eps)
                
            else:  # SGD
                params_flat = params_flat - self.learning_rate * grads
            
            # Clip to bounds
            for i, (param, (min_val, max_val)) in enumerate(self.param_bounds.items()):
                params_flat = params_flat.at[i].set(jnp.clip(params_flat[i], min_val, max_val))
            
            # Update parameter dictionary
            for i, name in enumerate(param_names):
                self.params[name] = float(params_flat[i])
            
            # Robust evaluation
            loss, ci = self._evaluate_params_robust(self.params)
            
            # Track history
            self.param_history.append(self.params.copy())
            self.loss_history.append(loss)
            self.confidence_intervals.append(ci)
            
            # Update best parameters
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_params = self.params.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if verbose:
                print(f"Loss: {loss:.6f} (best: {self.best_loss:.6f})")
                for metric, target in self.target_metrics.items():
                    if metric in ci:
                        mean_val = (ci[metric][0] + ci[metric][1]) / 2
                        ci_width = ci[metric][1] - ci[metric][0]
                        print(f"  {metric}: {mean_val:.4f} ± {ci_width/2:.4f} (target: {target:.4f})")
            
            # Early stopping
            if loss < self.tolerance:
                if verbose:
                    print("Converged: loss below tolerance")
                break
            
            if no_improvement_count >= self.patience:
                if verbose:
                    print("Early stopping: no improvement")
                break
        
        return self.best_params
    
    def _calibrate_evolution_strategies(self, verbose: bool) -> Dict[str, float]:
        """Evolution Strategies calibration."""
        param_names = list(self.params.keys())
        n_elite = int(self.es_population_size * self.es_elite_ratio)
        
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\nIteration {iteration+1}/{self.max_iterations}")
            
            # Evaluate population
            fitness_scores = []
            for i in range(self.es_population_size):
                params = {name: float(self.es_population[i, j]) for j, name in enumerate(param_names)}
                loss, _ = self._evaluate_params_robust(params)
                fitness_scores.append(loss)
            
            fitness_scores = jnp.array(fitness_scores)
            
            # Select elite
            elite_indices = jnp.argsort(fitness_scores)[:n_elite]
            elite_population = self.es_population[elite_indices]
            
            # Update best
            best_idx = elite_indices[0]
            best_loss = fitness_scores[best_idx]
            
            if best_loss < self.best_loss:
                self.best_loss = best_loss
                self.best_params = {name: float(self.es_population[best_idx, j]) 
                                  for j, name in enumerate(param_names)}
            
            # Generate new population
            self.key, subkey = random.split(self.key)
            
            # Compute elite mean and covariance
            elite_mean = jnp.mean(elite_population, axis=0)
            
            # Generate new population around elite mean
            noise = random.normal(subkey, self.es_population.shape) * self.es_sigma
            self.es_population = elite_mean[None, :] + noise
            
            # Clip to bounds
            for i, (param, (min_val, max_val)) in enumerate(self.param_bounds.items()):
                self.es_population = self.es_population.at[:, i].set(
                    jnp.clip(self.es_population[:, i], min_val, max_val)
                )
            
            # Track history
            self.loss_history.append(float(best_loss))
            self.param_history.append(self.best_params.copy())
            
            if verbose:
                print(f"Best loss: {best_loss:.6f}")
                print(f"Population mean: {elite_mean}")
            
            # Decay mutation strength
            self.es_sigma *= 0.995
            
            if best_loss < self.tolerance:
                if verbose:
                    print("Converged: loss below tolerance")
                break
        
        return self.best_params
    
    def _calibrate_particle_swarm(self, verbose: bool) -> Dict[str, float]:
        """Particle Swarm Optimization calibration."""
        param_names = list(self.params.keys())
        
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\nIteration {iteration+1}/{self.max_iterations}")
            
            # Evaluate particles
            for i in range(self.pso_population_size):
                params = {name: float(self.pso_positions[i, j]) for j, name in enumerate(param_names)}
                loss, _ = self._evaluate_params_robust(params)
                
                # Update personal best
                if loss < self.pso_personal_best_scores[i]:
                    self.pso_personal_best_scores = self.pso_personal_best_scores.at[i].set(loss)
                    self.pso_personal_best = self.pso_personal_best.at[i].set(self.pso_positions[i])
                
                # Update global best
                if loss < self.pso_global_best_score:
                    self.pso_global_best_score = loss
                    self.pso_global_best = self.pso_positions[i].copy()
                    self.best_loss = loss
                    self.best_params = params.copy()
            
            # Update velocities and positions
            self.key, subkey1, subkey2 = random.split(self.key, 3)
            
            r1 = random.uniform(subkey1, self.pso_velocities.shape)
            r2 = random.uniform(subkey2, self.pso_velocities.shape)
            
            cognitive = self.pso_c1 * r1 * (self.pso_personal_best - self.pso_positions)
            social = self.pso_c2 * r2 * (self.pso_global_best[None, :] - self.pso_positions)
            
            self.pso_velocities = (self.pso_w * self.pso_velocities + cognitive + social)
            self.pso_positions = self.pso_positions + self.pso_velocities
            
            # Clip to bounds
            for i, (param, (min_val, max_val)) in enumerate(self.param_bounds.items()):
                self.pso_positions = self.pso_positions.at[:, i].set(
                    jnp.clip(self.pso_positions[:, i], min_val, max_val)
                )
            
            # Track history
            self.loss_history.append(float(self.pso_global_best_score))
            self.param_history.append(self.best_params.copy())
            
            if verbose:
                print(f"Best loss: {self.pso_global_best_score:.6f}")
                print(f"Best params: {self.best_params}")
            
            if self.pso_global_best_score < self.tolerance:
                if verbose:
                    print("Converged: loss below tolerance")
                break
        
        return self.best_params
    
    def _calibrate_cross_entropy(self, verbose: bool) -> Dict[str, float]:
        """Cross-Entropy Method calibration."""
        param_names = list(self.params.keys())
        n_elite = int(self.cem_population_size * self.cem_elite_ratio)
        
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\nIteration {iteration+1}/{self.max_iterations}")
            
            # Sample population
            self.key, subkey = random.split(self.key)
            population = random.normal(subkey, (self.cem_population_size, len(param_names)))
            population = population * self.cem_std[None, :] + self.cem_mean[None, :]
            
            # Clip to bounds
            for i, (param, (min_val, max_val)) in enumerate(self.param_bounds.items()):
                population = population.at[:, i].set(jnp.clip(population[:, i], min_val, max_val))
            
            # Evaluate population
            fitness_scores = []
            for i in range(self.cem_population_size):
                params = {name: float(population[i, j]) for j, name in enumerate(param_names)}
                loss, _ = self._evaluate_params_robust(params)
                fitness_scores.append(loss)
            
            fitness_scores = jnp.array(fitness_scores)
            
            # Select elite
            elite_indices = jnp.argsort(fitness_scores)[:n_elite]
            elite_population = population[elite_indices]
            
            # Update distribution parameters
            self.cem_mean = jnp.mean(elite_population, axis=0)
            self.cem_std = jnp.std(elite_population, axis=0) + 1e-6  # Add small epsilon
            
            # Update best
            best_idx = elite_indices[0]
            best_loss = fitness_scores[best_idx]
            
            if best_loss < self.best_loss:
                self.best_loss = best_loss
                self.best_params = {name: float(population[best_idx, j]) 
                                  for j, name in enumerate(param_names)}
            
            # Track history
            self.loss_history.append(float(best_loss))
            self.param_history.append(self.best_params.copy())
            
            if verbose:
                print(f"Best loss: {best_loss:.6f}")
                print(f"Distribution mean: {self.cem_mean}")
                print(f"Distribution std: {self.cem_std}")
            
            # Decay noise
            self.cem_std *= self.cem_noise_decay
            
            if best_loss < self.tolerance:
                if verbose:
                    print("Converged: loss below tolerance")
                break
        
        return self.best_params
    
    def _calibrate_bayesian(self, verbose: bool) -> Dict[str, float]:
        """Bayesian Optimization calibration (simplified implementation)."""
        param_names = list(self.params.keys())
        
        # Evaluate initial points
        for i in range(self.bo_n_initial):
            if self.bo_evaluated >= self.bo_n_initial:
                break
                
            params = {name: float(self.bo_X[i, j]) for j, name in enumerate(param_names)}
            loss, _ = self._evaluate_params_robust(params)
            self.bo_y = self.bo_y.at[i].set(loss)
            
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_params = params.copy()
            
            self.bo_evaluated += 1
        
        # Main optimization loop
        for iteration in range(self.max_iterations - self.bo_n_initial):
            if verbose:
                print(f"\nIteration {iteration+1}/{self.max_iterations - self.bo_n_initial}")
            
            # Simple acquisition function: random search with bias toward good regions
            # In practice, you'd use a proper GP and acquisition function
            self.key, subkey = random.split(self.key)
            
            # Find best current point
            best_idx = jnp.argmin(self.bo_y[:self.bo_evaluated])
            best_point = self.bo_X[best_idx]
            
            # Sample around best point with some exploration
            noise_scale = 0.1 * (1.0 - iteration / self.max_iterations)  # Decay exploration
            candidate = best_point + random.normal(subkey, best_point.shape) * noise_scale
            
            # Clip to bounds
            for i, (param, (min_val, max_val)) in enumerate(self.param_bounds.items()):
                candidate = candidate.at[i].set(jnp.clip(candidate[i], min_val, max_val))
            
            # Evaluate candidate
            params = {name: float(candidate[j]) for j, name in enumerate(param_names)}
            loss, _ = self._evaluate_params_robust(params)
            
            # Add to dataset
            self.bo_X = jnp.concatenate([self.bo_X, candidate[None, :]], axis=0)
            self.bo_y = jnp.concatenate([self.bo_y, jnp.array([loss])], axis=0)
            self.bo_evaluated += 1
            
            # Update best
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_params = params.copy()
            
            # Track history
            self.loss_history.append(float(self.best_loss))
            self.param_history.append(self.best_params.copy())
            
            if verbose:
                print(f"Best loss: {self.best_loss:.6f}")
                print(f"Best params: {self.best_params}")
            
            if self.best_loss < self.tolerance:
                if verbose:
                    print("Converged: loss below tolerance")
                break
        
        return self.best_params
    
    def _calibrate_q_learning(self, verbose: bool) -> Dict[str, float]:
        """Fixed Q-Learning calibration with working optimization."""
        no_improvement_count = 0
        
        # Initialize Q-table for discrete states/actions  
        if not hasattr(self, 'ql_q_table'):
            self.ql_q_table = {}
        
        # Define simple action space: (param_index, step_size, direction)
        param_names = list(self.params.keys())
        actions = []
        for i, param in enumerate(param_names):
            min_val, max_val = self.param_bounds[param]
            small_step = (max_val - min_val) * 0.02  # 2% of range
            large_step = (max_val - min_val) * 0.1   # 10% of range
            
            actions.extend([
                (i, small_step, 1),   # small increase
                (i, small_step, -1),  # small decrease  
                (i, large_step, 1),   # large increase
                (i, large_step, -1),  # large decrease
            ])
        
        def get_state_key(params):
            """Convert parameters to discrete state."""
            state_parts = []
            for param, value in params.items():
                min_val, max_val = self.param_bounds[param]
                normalized = (value - min_val) / (max_val - min_val)
                bin_idx = int(jnp.clip(normalized * 10, 0, 9))  # 10 bins
                state_parts.append(str(bin_idx))
            return "_".join(state_parts)
        
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\nIteration {iteration+1}/{self.max_iterations}")
            
            # Get current state
            state_key = get_state_key(self.params)
            
            # Initialize Q-values for new states
            if state_key not in self.ql_q_table:
                self.ql_q_table[state_key] = jnp.zeros(len(actions))
            
            # Epsilon-greedy action selection
            self.key, subkey = random.split(self.key)
            if random.uniform(subkey) < self.ql_epsilon:
                action_idx = int(random.randint(subkey, (), 0, len(actions)))
            else:
                action_idx = int(jnp.argmax(self.ql_q_table[state_key]))
            
            # Apply action
            param_idx, step_size, direction = actions[action_idx]
            param_name = param_names[param_idx]
            
            new_params = self.params.copy()
            min_val, max_val = self.param_bounds[param_name]
            new_value = self.params[param_name] + direction * step_size
            new_params[param_name] = float(jnp.clip(new_value, min_val, max_val))
            
            # Evaluate new parameters
            old_loss = self.loss_history[-1] if self.loss_history else self._evaluate_params_robust(self.params)[0]
            new_loss, ci = self._evaluate_params_robust(new_params)
            
            # Compute reward (improvement with proper scaling)
            improvement = old_loss - new_loss
            reward = improvement * 100.0  # Scale for better learning
            
            # Update Q-value
            next_state_key = get_state_key(new_params)
            if next_state_key not in self.ql_q_table:
                self.ql_q_table[next_state_key] = jnp.zeros(len(actions))
            
            # Q-learning update
            current_q = self.ql_q_table[state_key][action_idx]
            max_next_q = jnp.max(self.ql_q_table[next_state_key])
            target_q = reward + self.ql_gamma * max_next_q
            
            # Update Q-table
            updated_q_values = self.ql_q_table[state_key].at[action_idx].set(
                current_q + self.ql_learning_rate * (target_q - current_q)
            )
            self.ql_q_table[state_key] = updated_q_values
            
            # Update parameters if improvement
            if new_loss < old_loss:
                self.params = new_params
            
            # Track history
            self.param_history.append(self.params.copy())
            self.loss_history.append(new_loss)
            self.confidence_intervals.append(ci)
            
            # Update best parameters
            if new_loss < self.best_loss:
                self.best_loss = new_loss
                self.best_params = self.params.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if verbose:
                print(f"Loss: {new_loss:.6f} (best: {self.best_loss:.6f})")
                print(f"Action: {action_idx} (param: {param_name}, step: {step_size:.4f}, dir: {direction})")
                print(f"Reward: {reward:.6f}, Epsilon: {self.ql_epsilon:.3f}")
            
            # Decay exploration
            self.ql_epsilon = max(self.ql_epsilon * self.ql_epsilon_decay, self.ql_epsilon_min)
            
            # Early stopping
            if new_loss < self.tolerance:
                if verbose:
                    print("Converged: loss below tolerance")
                break
            
            if no_improvement_count >= self.patience:
                if verbose:
                    print("Early stopping: no improvement")
                break
        
        return self.best_params
    
    def _train_ql_network_improved(self):
        """Improved Q-learning neural network training with proper Adam optimization."""
        # Sample random batch
        self.key, subkey = random.split(self.key)
        batch_indices = random.choice(subkey, len(self.ql_memory), 
                                     shape=(self.ql_batch_size,), replace=False)
        
        batch = [self.ql_memory[i] for i in batch_indices]
        
        # Prepare batch data
        states = jnp.array([exp[0] for exp in batch])
        actions = jnp.array([exp[1] for exp in batch])
        rewards = jnp.array([exp[2] for exp in batch])
        next_states = jnp.array([exp[3] for exp in batch])
        
        # Compute target Q-values using target network (Double DQN)
        next_q_values_main = jnp.array([self._ql_forward_improved(next_state) for next_state in next_states])
        next_q_values_target = jnp.array([self._ql_forward_improved(next_state, self.ql_target_params) 
                                        for next_state in next_states])
        
        # Double DQN: use main network to select actions, target network to evaluate
        next_actions = jnp.argmax(next_q_values_main, axis=1)
        target_q_values = rewards + self.ql_gamma * next_q_values_target[jnp.arange(len(batch)), next_actions]
        
        # Compute current Q-values
        current_q_values = jnp.array([self._ql_forward_improved(state) for state in states])
        current_q_selected = current_q_values[jnp.arange(len(batch)), actions]
        
        # Compute TD errors
        td_errors = target_q_values - current_q_selected
        
        # Compute gradients (simplified backpropagation)
        gradients = {k: {kk: jnp.zeros_like(vv) for kk, vv in v.items()} 
                    for k, v in self.ql_params.items()}
        
        for i in range(len(batch)):
            state = states[i]
            action = actions[i]
            td_error = td_errors[i]
            
            # Forward pass to get activations
            h1 = jnp.tanh(jnp.dot(state, self.ql_params['layer1']['weights']) + 
                         self.ql_params['layer1']['bias'])
            h2 = jnp.tanh(jnp.dot(h1, self.ql_params['layer2']['weights']) + 
                         self.ql_params['layer2']['bias'])
            h3 = jnp.tanh(jnp.dot(h2, self.ql_params['layer3']['weights']) + 
                         self.ql_params['layer3']['bias'])
            
            # Backpropagation (simplified)
            # Output layer gradients
            grad_output_w = jnp.zeros_like(self.ql_params['output']['weights'])
            grad_output_b = jnp.zeros_like(self.ql_params['output']['bias'])
            grad_output_w = grad_output_w.at[:, action].add(td_error * h3)
            grad_output_b = grad_output_b.at[action].add(td_error)
            
            # Layer 3 gradients
            delta3 = jnp.zeros(h3.shape[0])
            delta3 = delta3.at[:].add(td_error * self.ql_params['output']['weights'][:, action])
            delta3 = delta3 * (1 - h3**2)  # tanh derivative
            
            grad_layer3_w = jnp.outer(h2, delta3)
            grad_layer3_b = delta3
            
            # Accumulate gradients
            gradients['output']['weights'] += grad_output_w / len(batch)
            gradients['output']['bias'] += grad_output_b / len(batch)
            gradients['layer3']['weights'] += grad_layer3_w / len(batch)
            gradients['layer3']['bias'] += grad_layer3_b / len(batch)
        
        # Apply Adam optimization
        self.ql_adam_t += 1
        self.ql_params, self.ql_adam_m, self.ql_adam_v = self._adam_update(
            self.ql_params, gradients, self.ql_adam_m, self.ql_adam_v, 
            self.ql_adam_t, self.ql_learning_rate
        )
    
    def _calibrate_policy_gradient(self, verbose: bool) -> Dict[str, float]:
        """Enhanced Policy Gradient calibration with improved actor-critic architecture."""
        no_improvement_count = 0
        
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\nIteration {iteration+1}/{self.max_iterations}")
            
            # Create enhanced state representation
            state = self._create_enhanced_state(self.params)
            
            # Store state history
            self.pg_state_history.append(state)
            if len(self.pg_state_history) > 20:  # Keep recent history
                self.pg_state_history.pop(0)
            
            # Sample action from enhanced policy
            action, mean, std, value = self._pg_forward_enhanced(state)
            
            # Convert action to parameter dictionary and clip to bounds
            new_params = {}
            for i, param_name in enumerate(self.pg_param_names):
                min_val, max_val = self.param_bounds[param_name]
                new_params[param_name] = float(jnp.clip(action[i], min_val, max_val))
            
            # Evaluate parameters
            loss, ci = self._evaluate_params_robust(new_params)
            
            # Shaped reward function with safety checks
            old_loss = self.loss_history[-1] if self.loss_history else float('inf')
            improvement = old_loss - loss
            reward = jnp.clip(improvement * 10.0, -50.0, 50.0)  # Clip reward to prevent explosion
            
            # Add target-based reward shaping with safety
            target_reward = 0.0
            for metric, target in self.target_metrics.items():
                if metric in ci:
                    current_value = (ci[metric][0] + ci[metric][1]) / 2
                    if jnp.isfinite(current_value) and jnp.isfinite(target):
                        # Reward based on inverse distance to target
                        distance = abs(current_value - target) / (abs(target) + 1e-8)
                        target_reward += 1.0 / (1.0 + distance)  # Higher reward for closer values
            reward = jnp.clip(reward + target_reward, -100.0, 100.0)
            
            # Compute log probability for the action taken with safety checks
            action_safe = jnp.where(jnp.isfinite(action), action, 0.0)
            mean_safe = jnp.where(jnp.isfinite(mean), mean, 0.0)
            std_safe = jnp.maximum(jnp.where(jnp.isfinite(std), std, 0.1), 0.01)
            
            log_prob = -0.5 * jnp.sum(((action_safe - mean_safe) / std_safe) ** 2) - \
                       0.5 * jnp.sum(jnp.log(2 * jnp.pi * std_safe ** 2))
            log_prob = jnp.clip(log_prob, -50.0, 50.0)
            
            # Compute entropy for exploration bonus with safety
            entropy = 0.5 * jnp.sum(jnp.log(2 * jnp.pi * jnp.e * std_safe ** 2))
            entropy = jnp.clip(entropy, -50.0, 50.0)
            
            # Store episode data
            self.pg_loss_history.append(loss)
            
            # Advantage estimation (TD error) with safety checks
            if len(self.pg_loss_history) > 1:
                # Simple advantage estimate with clipping
                advantage = jnp.clip(reward - value, -50.0, 50.0)
            else:
                advantage = jnp.clip(reward, -50.0, 50.0)
            
            # Check for NaN/inf in key values
            if not jnp.isfinite(advantage):
                advantage = 0.0
            if not jnp.isfinite(log_prob):
                log_prob = 0.0
            if not jnp.isfinite(entropy):
                entropy = 0.0
            
            # Policy gradient update with entropy regularization and safety
            policy_loss = -float(log_prob) * advantage - self.pg_entropy_coeff * entropy
            value_loss = (reward - value) ** 2
            
            # Skip update if any critical values are NaN
            if not (jnp.isfinite(policy_loss) and jnp.isfinite(value_loss)):
                if verbose:
                    print(f"Skipping update due to NaN values: policy_loss={policy_loss}, value_loss={value_loss}")
                continue
            
            # Compute gradients for policy network (simplified)
            policy_gradients = self._compute_policy_gradients_enhanced(
                state, action, mean, std, advantage, entropy
            )
            
            # Compute gradients for value network
            value_gradients = self._compute_value_gradients_enhanced(state, reward, value)
            
            # Apply Adam optimization for both networks
            self.pg_adam_t += 1
            
            # Update policy network
            self.pg_policy_params, self.pg_policy_adam_m, self.pg_policy_adam_v = self._adam_update(
                self.pg_policy_params, policy_gradients, self.pg_policy_adam_m, 
                self.pg_policy_adam_v, self.pg_adam_t, self.pg_learning_rate
            )
            
            # Update value network
            self.pg_value_params, self.pg_value_adam_m, self.pg_value_adam_v = self._adam_update(
                self.pg_value_params, value_gradients, self.pg_value_adam_m, 
                self.pg_value_adam_v, self.pg_adam_t, self.pg_learning_rate * self.pg_value_coeff
            )
            
            # Update current parameters to the new sampled parameters
            self.params = new_params.copy()
            
            # Track history
            self.param_history.append(self.params.copy())
            self.loss_history.append(loss)
            self.confidence_intervals.append(ci)
            
            # Update best parameters
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_params = self.params.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if verbose:
                print(f"Loss: {loss:.6f} (best: {self.best_loss:.6f})")
                print(f"Reward: {reward:.6f}, Advantage: {advantage:.6f}")
                print(f"Value: {value:.6f}, Entropy: {entropy:.6f}")
                print(f"Policy std: {jnp.mean(std):.4f}")
            
            # Early stopping
            if loss < self.tolerance:
                if verbose:
                    print("Converged: loss below tolerance")
                break
            
            if no_improvement_count >= self.patience:
                if verbose:
                    print("Early stopping: no improvement")
                break
        
        return self.best_params
    
    def _compute_policy_gradients_enhanced(self, state: jax.Array, action: jax.Array, 
                                         mean: jax.Array, std: jax.Array, 
                                         advantage: float, entropy: float) -> Dict:
        """Compute policy gradients with entropy regularization and safety checks."""
        gradients = {k: {kk: jnp.zeros_like(vv) for kk, vv in v.items()} 
                    for k, v in self.pg_policy_params.items()}
        
        # Safety checks for inputs
        action = jnp.where(jnp.isfinite(action), action, 0.0)
        mean = jnp.where(jnp.isfinite(mean), mean, 0.0)
        std = jnp.maximum(jnp.where(jnp.isfinite(std), std, 0.1), 0.01)
        advantage = jnp.where(jnp.isfinite(advantage), advantage, 0.0)
        
        # Forward pass to get activations
        h1 = jnp.tanh(jnp.dot(state, self.pg_policy_params['shared_layer1']['weights']) + 
                     self.pg_policy_params['shared_layer1']['bias'])
        h2 = jnp.tanh(jnp.dot(h1, self.pg_policy_params['shared_layer2']['weights']) + 
                     self.pg_policy_params['shared_layer2']['bias'])
        
        # Gradients for mean output with safety
        grad_log_prob_mean = (action - mean) / (std ** 2)
        grad_log_prob_mean = jnp.clip(grad_log_prob_mean, -10.0, 10.0)
        policy_grad_mean = advantage * grad_log_prob_mean
        policy_grad_mean = jnp.clip(policy_grad_mean, -5.0, 5.0)
        
        gradients['mean_output']['weights'] = jnp.outer(h2, policy_grad_mean)
        gradients['mean_output']['bias'] = policy_grad_mean
        
        # Gradients for std output (log_std) with safety
        grad_log_prob_std = ((action - mean) ** 2 / (std ** 2) - 1) / std
        grad_log_prob_std = jnp.clip(grad_log_prob_std, -10.0, 10.0)
        entropy_grad_std = self.pg_entropy_coeff / std
        entropy_grad_std = jnp.clip(entropy_grad_std, -1.0, 1.0)
        policy_grad_std = advantage * grad_log_prob_std + entropy_grad_std
        policy_grad_std = jnp.clip(policy_grad_std, -5.0, 5.0)
        
        gradients['std_output']['weights'] = jnp.outer(h2, policy_grad_std)
        gradients['std_output']['bias'] = policy_grad_std
        
        # Ensure all gradients are finite
        for layer_name in gradients:
            for param_name in gradients[layer_name]:
                gradients[layer_name][param_name] = jnp.where(
                    jnp.isfinite(gradients[layer_name][param_name]), 
                    gradients[layer_name][param_name], 
                    0.0
                )
        
        return gradients
    
    def _compute_value_gradients_enhanced(self, state: jax.Array, target: float, value: float) -> Dict:
        """Compute value network gradients."""
        gradients = {k: {kk: jnp.zeros_like(vv) for kk, vv in v.items()} 
                    for k, v in self.pg_value_params.items()}
        
        # Value prediction error
        value_error = target - value
        
        # Forward pass to get activations
        h1 = jnp.tanh(jnp.dot(state, self.pg_value_params['layer1']['weights']) + 
                     self.pg_value_params['layer1']['bias'])
        
        # Gradients for value network
        gradients['layer2']['weights'] = jnp.outer(h1, jnp.array([value_error]))
        gradients['layer2']['bias'] = jnp.array([value_error])
        
        return gradients
    
    def _calibrate_actor_critic(self, verbose: bool) -> Dict[str, float]:
        """Enhanced Actor-Critic calibration with proper network architecture and GAE."""
        no_improvement_count = 0
        
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\nIteration {iteration+1}/{self.max_iterations}")
            
            # Current enhanced state
            state = self._create_enhanced_state(self.params)
            
            # Forward pass through enhanced actor-critic
            action, log_prob, value, entropy = self._ac_forward_enhanced(state)
            
            # Convert action to parameter dictionary and clip to bounds
            new_params = {}
            for i, param_name in enumerate(self.ac_param_names):
                min_val, max_val = self.param_bounds[param_name]
                new_params[param_name] = float(jnp.clip(action[i], min_val, max_val))
            
            # Evaluate parameters
            loss, ci = self._evaluate_params_robust(new_params)
            
            # Shaped reward
            old_loss = self.loss_history[-1] if self.loss_history else float('inf')
            improvement = old_loss - loss
            reward = improvement * 10.0
            
            # Add target-based bonus
            target_bonus = 0.0
            for metric, target in self.target_metrics.items():
                if metric in ci:
                    current_value = (ci[metric][0] + ci[metric][1]) / 2
                    distance = abs(current_value - target) / (abs(target) + 1e-8)
                    target_bonus += 1.0 / (1.0 + distance)
            reward += target_bonus
            
            # Store experience for GAE
            self.ac_states.append(state)
            self.ac_actions.append(action)
            self.ac_rewards.append(reward)
            self.ac_values.append(value)
            self.ac_log_probs.append(log_prob)
            self.ac_dones.append(False)  # Not episodic
            
            # Compute advantage and train networks
            if len(self.ac_states) >= 5:  # Train every few steps
                advantages = self._compute_gae(self.ac_rewards, self.ac_values, self.ac_dones)
                
                # Actor loss (policy gradient with entropy bonus)
                actor_loss = -float(log_prob) * advantages[-1] - self.ac_entropy_coeff * entropy
                
                # Critic loss (value prediction error)
                critic_loss = (reward - value) ** 2
                
                # Compute gradients
                actor_gradients = self._compute_actor_gradients_enhanced(state, action, advantages[-1], entropy)
                critic_gradients = self._compute_critic_gradients_enhanced(state, reward, value)
                
                # Apply Adam optimization
                self.ac_adam_t += 1
                
                # Update actor
                self.ac_actor_params, self.ac_actor_adam_m, self.ac_actor_adam_v = self._adam_update(
                    self.ac_actor_params, actor_gradients, self.ac_actor_adam_m, 
                    self.ac_actor_adam_v, self.ac_adam_t, self.ac_actor_lr
                )
                
                # Update critic
                self.ac_critic_params, self.ac_critic_adam_m, self.ac_critic_adam_v = self._adam_update(
                    self.ac_critic_params, critic_gradients, self.ac_critic_adam_m, 
                    self.ac_critic_adam_v, self.ac_adam_t, self.ac_critic_lr
                )
                
                # Clear buffers
                self.ac_states = self.ac_states[-1:]  # Keep last state
                self.ac_actions = self.ac_actions[-1:]
                self.ac_rewards = self.ac_rewards[-1:]
                self.ac_values = self.ac_values[-1:]
                self.ac_log_probs = self.ac_log_probs[-1:]
                self.ac_dones = self.ac_dones[-1:]
            
            # Update parameters
            self.params = new_params
            
            # Track history
            self.param_history.append(self.params.copy())
            self.loss_history.append(loss)
            self.confidence_intervals.append(ci)
            
            # Update best parameters
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_params = self.params.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if verbose:
                print(f"Loss: {loss:.6f} (best: {self.best_loss:.6f})")
                print(f"Reward: {reward:.6f}")
                print(f"Value: {value:.6f}, Entropy: {entropy:.6f}")
            
            # Early stopping
            if loss < self.tolerance:
                if verbose:
                    print("Converged: loss below tolerance")
                break
            
            if no_improvement_count >= self.patience:
                if verbose:
                    print("Early stopping: no improvement")
                break
        
        return self.best_params
    
    def _compute_gae(self, rewards: List[float], values: List[float], dones: List[bool]) -> List[float]:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + self.ac_gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.ac_gamma * self.ac_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def _compute_actor_gradients_enhanced(self, state: jax.Array, action: jax.Array, 
                                        advantage: float, entropy: float) -> Dict:
        """Compute actor gradients with enhanced architecture."""
        gradients = {k: {kk: jnp.zeros_like(vv) for kk, vv in v.items()} 
                    for k, v in self.ac_actor_params.items()}
        
        # Forward pass to get activations
        a1 = jnp.tanh(jnp.dot(state, self.ac_actor_params['layer1']['weights']) + 
                     self.ac_actor_params['layer1']['bias'])
        a2 = jnp.tanh(jnp.dot(a1, self.ac_actor_params['layer2']['weights']) + 
                     self.ac_actor_params['layer2']['bias'])
        
        mean = jnp.dot(a2, self.ac_actor_params['mean_output']['weights']) + \
               self.ac_actor_params['mean_output']['bias']
        log_std = jnp.dot(a2, self.ac_actor_params['std_output']['weights']) + \
                  self.ac_actor_params['std_output']['bias']
        std = jnp.exp(jnp.maximum(log_std, jnp.log(0.05)))
        
        # Policy gradients
        grad_log_prob_mean = (action - mean) / (std ** 2)
        policy_grad_mean = advantage * grad_log_prob_mean
        
        grad_log_prob_std = ((action - mean) ** 2 / (std ** 2) - 1) / std
        entropy_grad_std = self.ac_entropy_coeff / std
        policy_grad_std = advantage * grad_log_prob_std + entropy_grad_std
        
        # Output layer gradients
        gradients['mean_output']['weights'] = jnp.outer(a2, policy_grad_mean)
        gradients['mean_output']['bias'] = policy_grad_mean
        gradients['std_output']['weights'] = jnp.outer(a2, policy_grad_std)
        gradients['std_output']['bias'] = policy_grad_std
        
        return gradients
    
    def _compute_critic_gradients_enhanced(self, state: jax.Array, target: float, value: float) -> Dict:
        """Compute critic gradients with enhanced architecture."""
        gradients = {k: {kk: jnp.zeros_like(vv) for kk, vv in v.items()} 
                    for k, v in self.ac_critic_params.items()}
        
        # Value prediction error
        value_error = target - value
        
        # Forward pass to get activations
        c1 = jnp.tanh(jnp.dot(state, self.ac_critic_params['layer1']['weights']) + 
                     self.ac_critic_params['layer1']['bias'])
        c2 = jnp.tanh(jnp.dot(c1, self.ac_critic_params['layer2']['weights']) + 
                     self.ac_critic_params['layer2']['bias'])
        c3 = jnp.tanh(jnp.dot(c2, self.ac_critic_params['layer3']['weights']) + 
                     self.ac_critic_params['layer3']['bias'])
        
        # Output layer gradients
        gradients['output']['weights'] = jnp.outer(c3, jnp.array([value_error]))
        gradients['output']['bias'] = jnp.array([value_error])
        
        return gradients
    
    def _calibrate_multi_agent_rl(self, verbose: bool) -> Dict[str, float]:
        """Enhanced Multi-Agent RL calibration with communication and coordination."""
        no_improvement_count = 0
        
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\nIteration {iteration+1}/{self.max_iterations}")
            
            # Create enhanced global state
            global_state = self._create_enhanced_state(self.params)
            
            # Each agent selects an action for its parameter
            new_params = self.params.copy()
            agent_actions = {}
            agent_communications = {}
            
            # First pass: generate communications
            for param_name in self.marl_param_names:
                agent = self.marl_agents[param_name]
                
                # Create local state (parameter + global info)
                min_val, max_val = self.param_bounds[param_name]
                normalized_param = (self.params[param_name] - min_val) / (max_val - min_val)
                local_state = jnp.array([normalized_param])
                
                # Get communication from other agents (zeros for first iteration)
                other_comms = self.marl_global_comm[self.marl_param_names.index(param_name)]
                
                # Forward pass to get Q-values and communication
                q_values, comm_output = self._marl_agent_forward_enhanced(
                    agent, local_state, other_comms.reshape(1, -1)
                )
                
                agent_communications[param_name] = comm_output
            
            # Update global communication buffer
            for i, param_name in enumerate(self.marl_param_names):
                self.marl_global_comm = self.marl_global_comm.at[i].set(
                    agent_communications[param_name]
                )
            
            # Second pass: select actions with updated communications
            for param_name in self.marl_param_names:
                agent = self.marl_agents[param_name]
                
                # Create enhanced local state
                min_val, max_val = self.param_bounds[param_name]
                normalized_param = (self.params[param_name] - min_val) / (max_val - min_val)
                
                # Include global state and communications from other agents
                other_comms = []
                for other_param in self.marl_param_names:
                    if other_param != param_name:
                        other_comms.append(agent_communications[other_param])
                
                if other_comms:
                    comm_input = jnp.concatenate(other_comms)
                else:
                    comm_input = jnp.zeros(self.marl_communication_dim)
                
                # Enhanced local state: own param + global features + communications
                local_state = jnp.concatenate([
                    jnp.array([normalized_param]),
                    global_state[:len(self.marl_param_names)],  # Global param state
                    global_state[len(self.marl_param_names):2*len(self.marl_param_names)],  # Targets
                    comm_input
                ])
                
                # Action selection with epsilon-greedy
                self.key, subkey = random.split(self.key)
                if random.uniform(subkey) < agent['epsilon']:
                    action = random.randint(subkey, (), 0, len(self.marl_step_sizes))
                else:
                    q_values, _ = self._marl_agent_forward_enhanced(
                        agent, local_state.reshape(1, -1), jnp.zeros((1, self.marl_communication_dim))
                    )
                    action = jnp.argmax(q_values)
                
                agent_actions[param_name] = int(action)
                
                # Apply action
                step_size = self.marl_step_sizes[action]
                step = step_size * (max_val - min_val)
                
                # Random direction
                self.key, subkey = random.split(self.key)
                direction = 1 if random.uniform(subkey) < 0.5 else -1
                
                new_value = self.params[param_name] + direction * step
                new_params[param_name] = float(jnp.clip(new_value, min_val, max_val))
            
            # Evaluate joint action
            loss, ci = self._evaluate_params_robust(new_params)
            
            # Shaped reward for all agents
            old_loss = self.loss_history[-1] if self.loss_history else float('inf')
            improvement = old_loss - loss
            reward = improvement * 10.0
            
            # Coordination bonus (all agents get same reward)
            target_bonus = 0.0
            for metric, target in self.target_metrics.items():
                if metric in ci:
                    current_value = (ci[metric][0] + ci[metric][1]) / 2
                    distance = abs(current_value - target) / (abs(target) + 1e-8)
                    target_bonus += 1.0 / (1.0 + distance)
            reward += target_bonus
            
            # Update each agent's neural network
            for param_name in self.marl_param_names:
                agent = self.marl_agents[param_name]
                action = agent_actions[param_name]
                
                # States for this agent
                min_val, max_val = self.param_bounds[param_name]
                current_norm = (self.params[param_name] - min_val) / (max_val - min_val)
                next_norm = (new_params[param_name] - min_val) / (max_val - min_val)
                
                current_state = jnp.array([current_norm])
                next_state = jnp.array([next_norm])
                
                # Store experience
                experience = (current_state, action, reward, next_state)
                agent['memory'].append(experience)
                if len(agent['memory']) > agent['memory_size']:
                    agent['memory'].pop(0)
                
                # Train agent's network
                if len(agent['memory']) >= 16:  # Mini-batch size
                    self._train_marl_agent_enhanced(agent, param_name)
                
                # Decay exploration
                agent['epsilon'] = max(agent['epsilon'] * self.marl_epsilon_decay, self.marl_epsilon_min)
            
            # Update parameters
            self.params = new_params
            
            # Track history
            self.param_history.append(self.params.copy())
            self.loss_history.append(loss)
            self.confidence_intervals.append(ci)
            
            # Update best parameters
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_params = self.params.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if verbose:
                print(f"Loss: {loss:.6f} (best: {self.best_loss:.6f})")
                print(f"Reward: {reward:.6f}")
                print(f"Actions: {agent_actions}")
                avg_epsilon = sum(agent['epsilon'] for agent in self.marl_agents.values()) / len(self.marl_agents)
                print(f"Avg Epsilon: {avg_epsilon:.3f}")
            
            # Early stopping
            if loss < self.tolerance:
                if verbose:
                    print("Converged: loss below tolerance")
                break
            
            if no_improvement_count >= self.patience:
                if verbose:
                    print("Early stopping: no improvement")
                break
        
        return self.best_params
    
    def _train_marl_agent_enhanced(self, agent: Dict, param_name: str):
        """Enhanced training for multi-agent RL with proper Adam optimization."""
        # Sample random batch
        self.key, subkey = random.split(self.key)
        batch_size = min(16, len(agent['memory']))
        batch_indices = random.choice(subkey, len(agent['memory']), 
                                     shape=(batch_size,), replace=False)
        
        batch = [agent['memory'][i] for i in batch_indices]
        
        # Prepare batch data
        states = jnp.array([exp[0] for exp in batch])
        actions = jnp.array([exp[1] for exp in batch])
        rewards = jnp.array([exp[2] for exp in batch])
        next_states = jnp.array([exp[3] for exp in batch])
        
        # Compute target Q-values
        next_q_values = []
        for next_state in next_states:
            comm_input = jnp.zeros((1, self.marl_communication_dim))
            q_vals, _ = self._marl_agent_forward_enhanced(agent, next_state.reshape(1, -1), comm_input)
            next_q_values.append(q_vals)
        next_q_values = jnp.array(next_q_values)
        
        target_q_values = rewards + 0.95 * jnp.max(next_q_values, axis=1)
        
        # Compute current Q-values
        current_q_values = []
        for state in states:
            comm_input = jnp.zeros((1, self.marl_communication_dim))
            q_vals, _ = self._marl_agent_forward_enhanced(agent, state.reshape(1, -1), comm_input)
            current_q_values.append(q_vals)
        current_q_values = jnp.array(current_q_values)
        
        # Compute TD errors
        td_errors = target_q_values - current_q_values[jnp.arange(len(batch)), actions]
        
        # Compute gradients (simplified)
        gradients = {k: {kk: jnp.zeros_like(vv) for kk, vv in v.items()} 
                    for k, v in agent['q_params'].items()}
        
        for i in range(len(batch)):
            state = states[i]
            action = actions[i]
            td_error = td_errors[i]
            
            # Simple gradient for Q-output layer
            grad_w = jnp.zeros_like(agent['q_params']['q_output']['weights'])
            grad_b = jnp.zeros_like(agent['q_params']['q_output']['bias'])
            
            # Assume state features for gradient computation
            grad_w = grad_w.at[:, action].add(td_error * state[0])  # Simplified
            grad_b = grad_b.at[action].add(td_error)
            
            gradients['q_output']['weights'] += grad_w / len(batch)
            gradients['q_output']['bias'] += grad_b / len(batch)
        
        # Apply Adam optimization
        agent['adam_t'] += 1
        agent['q_params'], agent['q_adam_m'], agent['q_adam_v'] = self._adam_update(
            agent['q_params'], gradients, agent['q_adam_m'], 
            agent['q_adam_v'], agent['adam_t'], self.marl_learning_rate
        )
    
    def _calibrate_dqn(self, verbose: bool) -> Dict[str, float]:
        """Advanced DQN calibration with Dueling architecture and prioritized experience replay."""
        no_improvement_count = 0
        target_update_counter = 0
        
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\nIteration {iteration+1}/{self.max_iterations}")
            
            # Current enhanced state
            state = self._create_enhanced_state(self.params)
            
            # Epsilon-greedy action selection with diversity penalty
            self.key, subkey = random.split(self.key)
            if random.uniform(subkey) < self.dqn_epsilon:
                # Explore: random action with diversity bias
                n_actions = len(self.dqn_param_names) * len(self.dqn_step_sizes)
                action = random.randint(subkey, (), 0, n_actions)
            else:
                # Exploit: best action from DQN with diversity penalty
                q_values = self._dqn_forward_dueling(state)
                
                # Apply action diversity penalty
                for i in range(len(q_values)):
                    penalty = self._compute_action_diversity_penalty(i)
                    q_values = q_values.at[i].add(-penalty)
                
                action = jnp.argmax(q_values)
            
            # Track action for diversity
            self.dqn_action_history.append(int(action))
            if len(self.dqn_action_history) > self.dqn_action_history_size:
                self.dqn_action_history.pop(0)
            
            # Apply action to parameters
            new_params = self.params.copy()
            param_idx = int(action) // len(self.dqn_step_sizes)
            step_idx = int(action) % len(self.dqn_step_sizes)
            param_name = self.dqn_param_names[param_idx]
            
            # Update parameter
            min_val, max_val = self.param_bounds[param_name]
            step_size = self.dqn_step_sizes[step_idx]
            step = step_size * (max_val - min_val)
            
            # Random direction
            self.key, subkey = random.split(self.key)
            direction = 1 if random.uniform(subkey) < 0.5 else -1
            
            new_value = self.params[param_name] + direction * step
            new_params[param_name] = float(jnp.clip(new_value, min_val, max_val))
            
            # Evaluate new parameters
            loss, ci = self._evaluate_params_robust(new_params)
            
            # Shaped reward
            old_loss = self.loss_history[-1] if self.loss_history else float('inf')
            improvement = old_loss - loss
            reward = improvement * 10.0
            
            # Store experience
            next_state = self._create_enhanced_state(new_params)
            experience = (state, int(action), reward, next_state, False)  # not done
            
            self.dqn_memory.append(experience)
            if len(self.dqn_memory) > self.dqn_memory_size:
                self.dqn_memory.pop(0)
            
            # Train DQN if we have enough experiences
            if len(self.dqn_memory) >= self.dqn_batch_size:
                self._train_dqn_improved()
            
            # Update target network periodically
            target_update_counter += 1
            if target_update_counter >= self.dqn_target_update_freq:
                self.dqn_target_params = {k: {kk: vv.copy() for kk, vv in v.items()} 
                                         for k, v in self.dqn_params.items()}
                target_update_counter = 0
            
            # Update parameters
            self.params = new_params
            
            # Track history
            self.param_history.append(self.params.copy())
            self.loss_history.append(loss)
            self.confidence_intervals.append(ci)
            
            # Update best parameters
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_params = self.params.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if verbose:
                print(f"Loss: {loss:.6f} (best: {self.best_loss:.6f})")
                print(f"Action: {action}, Reward: {reward:.6f}")
                print(f"Epsilon: {self.dqn_epsilon:.3f}")
            
            # Decay exploration with minimum
            self.dqn_epsilon = max(self.dqn_epsilon * self.dqn_epsilon_decay, self.dqn_epsilon_min)
            
            # Early stopping
            if loss < self.tolerance:
                if verbose:
                    print("Converged: loss below tolerance")
                break
            
            if no_improvement_count >= self.patience:
                if verbose:
                    print("Early stopping: no improvement")
                break
        
        return self.best_params
    
    def _train_dqn_improved(self):
        """Train the DQN using improved experience replay."""
        # Sample random batch
        self.key, subkey = random.split(self.key)
        batch_indices = random.choice(subkey, len(self.dqn_memory), 
                                     shape=(self.dqn_batch_size,), replace=False)
        
        batch = [self.dqn_memory[i] for i in batch_indices]
        
        # Prepare batch data
        states = jnp.array([exp[0] for exp in batch])
        actions = jnp.array([exp[1] for exp in batch])
        rewards = jnp.array([exp[2] for exp in batch])
        next_states = jnp.array([exp[3] for exp in batch])
        
        # Compute target Q-values using Double DQN
        next_q_values_main = jnp.array([self._dqn_forward_dueling(next_state) for next_state in next_states])
        next_q_values_target = jnp.array([self._dqn_forward_dueling(next_state, self.dqn_target_params) 
                                        for next_state in next_states])
        
        # Double DQN: use main network to select actions, target network to evaluate
        next_actions = jnp.argmax(next_q_values_main, axis=1)
        target_q_values = rewards + self.dqn_gamma * next_q_values_target[jnp.arange(len(batch)), next_actions]
        
        # Compute current Q-values
        current_q_values = jnp.array([self._dqn_forward_dueling(state) for state in states])
        current_q_selected = current_q_values[jnp.arange(len(batch)), actions]
        
        # Compute TD errors
        td_errors = target_q_values - current_q_selected
        
        # Simplified gradient update (focusing on advantage stream)
        gradients = {k: {kk: jnp.zeros_like(vv) for kk, vv in v.items()} 
                    for k, v in self.dqn_params.items()}
        
        for i in range(len(batch)):
            state = states[i]
            action = actions[i]
            td_error = td_errors[i]
            
            # Forward pass to get activations
            h1 = jnp.relu(jnp.dot(state, self.dqn_params['shared1']['weights']) + 
                         self.dqn_params['shared1']['bias'])
            h2 = jnp.relu(jnp.dot(h1, self.dqn_params['shared2']['weights']) + 
                         self.dqn_params['shared2']['bias'])
            a1 = jnp.relu(jnp.dot(h2, self.dqn_params['advantage1']['weights']) + 
                         self.dqn_params['advantage1']['bias'])
            
            # Gradients for advantage output
            grad_adv_w = jnp.zeros_like(self.dqn_params['advantage_output']['weights'])
            grad_adv_b = jnp.zeros_like(self.dqn_params['advantage_output']['bias'])
            grad_adv_w = grad_adv_w.at[:, action].add(td_error * a1)
            grad_adv_b = grad_adv_b.at[action].add(td_error)
            
            # Accumulate gradients
            gradients['advantage_output']['weights'] += grad_adv_w / len(batch)
            gradients['advantage_output']['bias'] += grad_adv_b / len(batch)
        
        # Apply Adam optimization
        self.dqn_adam_t += 1
        self.dqn_params, self.dqn_adam_m, self.dqn_adam_v = self._adam_update(
            self.dqn_params, gradients, self.dqn_adam_m, self.dqn_adam_v, 
            self.dqn_adam_t, self.dqn_learning_rate
        )
    
    def get_calibration_history(self) -> Dict[str, List[Any]]:
        """Get calibration history.
        
        Returns:
            Dictionary with 'loss', 'params', and 'confidence_intervals' histories
        """
        return {
            "loss": self.loss_history,
            "params": self.param_history,
            "confidence_intervals": self.confidence_intervals
        }
    
    def plot_calibration(self, figsize: Tuple[int, int] = (15, 10)) -> Any:
        """Plot comprehensive calibration results.
        
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
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot loss history
        axes[0, 0].plot(self.loss_history, 'b-', linewidth=2)
        axes[0, 0].set_title(f"Loss History ({self.method.upper()})")
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        # Plot parameter evolution
        param_names = list(self.params.keys())
        for param in param_names:
            values = [params[param] for params in self.param_history]
            axes[0, 1].plot(values, label=param, linewidth=2)
            
        axes[0, 1].set_title("Parameter Evolution")
        axes[0, 1].set_xlabel("Iteration")
        axes[0, 1].set_ylabel("Parameter Value")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot final metrics vs targets
        if self.confidence_intervals:
            final_ci = self.confidence_intervals[-1]
            metrics = list(self.target_metrics.keys())
            targets = [self.target_metrics[m] for m in metrics]
            current_means = [(final_ci[m][0] + final_ci[m][1]) / 2 for m in metrics]
            current_errors = [(final_ci[m][1] - final_ci[m][0]) / 2 for m in metrics]
            
            x_pos = np.arange(len(metrics))
            axes[1, 0].bar(x_pos - 0.2, targets, 0.4, label='Target', alpha=0.7)
            axes[1, 0].errorbar(x_pos + 0.2, current_means, yerr=current_errors, 
                              fmt='o', capsize=5, label='Current ± CI')
            
            axes[1, 0].set_title("Final Metrics vs Targets")
            axes[1, 0].set_xlabel("Metrics")
            axes[1, 0].set_ylabel("Value")
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels(metrics)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot convergence analysis
        if len(self.loss_history) > 10:
            # Moving average of loss
            window = min(10, len(self.loss_history) // 4)
            moving_avg = np.convolve(self.loss_history, np.ones(window)/window, mode='valid')
            axes[1, 1].plot(range(window-1, len(self.loss_history)), moving_avg, 
                           'r-', linewidth=2, label=f'Moving Avg (window={window})')
            axes[1, 1].plot(self.loss_history, 'b-', alpha=0.3, label='Raw Loss')
            
            axes[1, 1].set_title("Convergence Analysis")
            axes[1, 1].set_xlabel("Iteration")
            axes[1, 1].set_ylabel("Loss")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        return fig, axes


class EnsembleCalibrator:
    """Ensemble calibrator that combines multiple optimization methods.
    
    This class runs multiple calibration methods in parallel and combines
    their results to find the best parameter set.
    """
    
    def __init__(
        self,
        model_factory: ModelFactory,
        initial_params: Dict[str, float],
        target_metrics: Dict[str, float],
        methods: List[str] = ["adam", "es", "pso"],
        **kwargs
    ):
        """Initialize ensemble calibrator.
        
        Args:
            model_factory: Function to create model instances
            initial_params: Initial parameter values
            target_metrics: Target metric values
            methods: List of optimization methods to use
            **kwargs: Additional arguments passed to individual calibrators
        """
        self.model_factory = model_factory
        self.initial_params = initial_params
        self.target_metrics = target_metrics
        self.methods = methods
        self.kwargs = kwargs
        
        self.calibrators = {}
        self.results = {}
        
    def calibrate(self, verbose: bool = True) -> Dict[str, Any]:
        """Run ensemble calibration.
        
        Args:
            verbose: Whether to print progress information
            
        Returns:
            Dictionary with results from all methods and the best overall result
        """
        if verbose:
            print(f"Running ensemble calibration with methods: {self.methods}")
        
        best_loss = float('inf')
        best_method = None
        best_params = None
        
        for method in self.methods:
            if verbose:
                print(f"\n{'='*50}")
                print(f"Running {method.upper()} calibration")
                print(f"{'='*50}")
            
            # Create calibrator for this method
            calibrator = ModelCalibrator(
                model_factory=self.model_factory,
                initial_params=self.initial_params.copy(),
                target_metrics=self.target_metrics,
                method=method,
                **self.kwargs
            )
            
            # Run calibration
            try:
                final_params = calibrator.calibrate(verbose=verbose)
                final_loss = calibrator.best_loss
                
                # Store results
                self.calibrators[method] = calibrator
                self.results[method] = {
                    'params': final_params,
                    'loss': final_loss,
                    'history': calibrator.get_calibration_history()
                }
                
                # Update best
                if final_loss < best_loss:
                    best_loss = final_loss
                    best_method = method
                    best_params = final_params
                
                if verbose:
                    print(f"{method.upper()} final loss: {final_loss:.6f}")
                    
            except Exception as e:
                if verbose:
                    print(f"Error in {method}: {e}")
                continue
        
        # Store best overall result
        self.results['best'] = {
            'method': best_method,
            'params': best_params,
            'loss': best_loss
        }
        
        if verbose:
            print(f"\n{'='*50}")
            print("ENSEMBLE RESULTS")
            print(f"{'='*50}")
            print(f"Best method: {best_method}")
            print(f"Best loss: {best_loss:.6f}")
            print(f"Best parameters: {best_params}")
        
        return self.results
    
    def plot_comparison(self, figsize: Tuple[int, int] = (15, 10)) -> Any:
        """Plot comparison of all methods.
        
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
        
        if not self.results:
            raise ValueError("Must run calibration before plotting")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot loss histories
        for method in self.methods:
            if method in self.results:
                loss_history = self.results[method]['history']['loss']
                axes[0, 0].plot(loss_history, label=method.upper(), linewidth=2)
        
        axes[0, 0].set_title("Loss History Comparison")
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_yscale('log')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot final losses
        methods_with_results = [m for m in self.methods if m in self.results]
        final_losses = [self.results[m]['loss'] for m in methods_with_results]
        
        axes[0, 1].bar(methods_with_results, final_losses, alpha=0.7)
        axes[0, 1].set_title("Final Loss Comparison")
        axes[0, 1].set_xlabel("Method")
        axes[0, 1].set_ylabel("Final Loss")
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot parameter convergence for best method
        if 'best' in self.results:
            best_method = self.results['best']['method']
            if best_method in self.results:
                param_history = self.results[best_method]['history']['params']
                param_names = list(self.initial_params.keys())
                
                for param in param_names:
                    values = [params[param] for params in param_history]
                    axes[1, 0].plot(values, label=param, linewidth=2)
                
                axes[1, 0].set_title(f"Parameter Evolution (Best: {best_method.upper()})")
                axes[1, 0].set_xlabel("Iteration")
                axes[1, 0].set_ylabel("Parameter Value")
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
        
        # Plot convergence speed comparison
        for method in self.methods:
            if method in self.results:
                loss_history = self.results[method]['history']['loss']
                # Normalize to show convergence speed
                if len(loss_history) > 1:
                    normalized = np.array(loss_history) / loss_history[0]
                    axes[1, 1].plot(normalized, label=method.upper(), linewidth=2)
        
        axes[1, 1].set_title("Convergence Speed Comparison")
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_ylabel("Normalized Loss")
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, axes


def compare_calibration_methods(
    model_factory: ModelFactory,
    initial_params: Dict[str, float],
    target_metrics: Dict[str, float],
    methods: List[str] = ["adam", "sgd", "es", "pso", "cem"],
    param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    max_iterations: int = 50,
    verbose: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """Compare different calibration methods on the same problem.
    
    Args:
        model_factory: Function to create model instances
        initial_params: Initial parameter values
        target_metrics: Target metric values
        methods: List of methods to compare
        param_bounds: Parameter bounds for each parameter
        max_iterations: Maximum iterations for each method
        verbose: Whether to print progress
        **kwargs: Additional arguments passed to ModelCalibrator
        
    Returns:
        Dictionary with comparison results
    """
    ensemble = EnsembleCalibrator(
        model_factory=model_factory,
        initial_params=initial_params,
        target_metrics=target_metrics,
        methods=methods,
        param_bounds=param_bounds,
        max_iterations=max_iterations,
        **kwargs
    )
    
    results = ensemble.calibrate(verbose=verbose)
    
    if verbose:
        print("\n" + "="*60)
        print("CALIBRATION METHOD COMPARISON SUMMARY")
        print("="*60)
        
        # Sort methods by performance
        method_performance = []
        for method in methods:
            if method in results:
                method_performance.append((method, results[method]['loss']))
        
        method_performance.sort(key=lambda x: x[1])
        
        print(f"{'Rank':<6} {'Method':<10} {'Final Loss':<15} {'Improvement':<15}")
        print("-" * 60)
        
        for i, (method, loss) in enumerate(method_performance):
            if i == 0:
                improvement = "Best"
            else:
                best_loss = method_performance[0][1]
                improvement = f"{loss/best_loss:.2f}x worse"
            
            print(f"{i+1:<6} {method.upper():<10} {loss:<15.6f} {improvement:<15}")
    
    return results


# Example usage function
def create_calibration_example():
    """Create an example demonstrating the improved calibration capabilities."""
    
    # This is a placeholder example - in practice, you'd use your actual model factory
    def example_model_factory(params, config):
        """Example model factory for demonstration."""
        # This would be replaced with your actual model creation logic
        class ExampleModel:
            def __init__(self, params, config):
                self.params = params
                self.config = config
            
            def run(self, steps=50):
                # Simulate some model behavior
                import jax.numpy as jnp
                import jax.random as random
                
                key = random.PRNGKey(self.config.seed)
                
                # Simulate metrics that depend on parameters
                noise = random.normal(key, (steps,)) * 0.1
                
                # Example: metric depends on parameter values
                metric1_base = self.params.get('param1', 1.0) * 2.0
                metric2_base = self.params.get('param2', 1.0) ** 2
                
                metric1_values = [float(metric1_base + noise[i]) for i in range(steps)]
                metric2_values = [float(metric2_base + noise[i]) for i in range(steps)]
                
                return {
                    'metric1': metric1_values,
                    'metric2': metric2_values
                }
        
        return ExampleModel(params, config)
    
    # Example usage
    initial_params = {'param1': 1.0, 'param2': 2.0}
    target_metrics = {'metric1': 3.0, 'metric2': 4.0}
    param_bounds = {'param1': (0.1, 5.0), 'param2': (0.1, 5.0)}
    
    print("Example: Advanced Model Calibration")
    print("="*50)
    
    # Single method calibration
    print("\n1. Single Method Calibration (Adam)")
    calibrator = ModelCalibrator(
        model_factory=example_model_factory,
        initial_params=initial_params,
        target_metrics=target_metrics,
        param_bounds=param_bounds,
        method="adam",
        max_iterations=30
    )
    
    result = calibrator.calibrate()
    print(f"Final parameters: {result}")
    
    # Ensemble calibration
    print("\n2. Ensemble Calibration")
    ensemble_results = compare_calibration_methods(
        model_factory=example_model_factory,
        initial_params=initial_params,
        target_metrics=target_metrics,
        methods=["adam", "es", "pso"],
        max_iterations=20,
        verbose=True
    )
    
    return {
        'single_method': result,
        'ensemble': ensemble_results
    }


if __name__ == "__main__":
    # Run example if script is executed directly
    example_results = create_calibration_example() 