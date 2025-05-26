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
        
        loss = self._compute_loss(mean_metrics)
        return loss, confidence_intervals
    
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
        """Set up Q-Learning for parameter optimization."""
        # Discretize parameter space for Q-learning
        self.ql_n_bins = 10  # Number of discrete bins per parameter
        self.ql_epsilon = 0.1  # Exploration rate
        self.ql_epsilon_decay = 0.995
        self.ql_alpha = 0.1  # Learning rate
        self.ql_gamma = 0.9  # Discount factor
        
        param_names = list(self.params.keys())
        n_params = len(param_names)
        
        # Create discrete state and action spaces
        self.ql_param_names = param_names
        self.ql_state_shape = tuple([self.ql_n_bins] * n_params)
        self.ql_n_actions = 2 * n_params  # +/- for each parameter
        
        # Initialize Q-table
        self.ql_q_table = jnp.zeros(self.ql_state_shape + (self.ql_n_actions,))
        
        # Parameter discretization bounds
        self.ql_param_bins = {}
        for param, (min_val, max_val) in self.param_bounds.items():
            self.ql_param_bins[param] = jnp.linspace(min_val, max_val, self.ql_n_bins)
        
        # Current state and step size
        self.ql_current_state = self._params_to_state(self.params)
        self.ql_step_size = 0.1  # Relative step size for actions
    
    def _setup_policy_gradient(self):
        """Set up Policy Gradient (REINFORCE) for parameter optimization."""
        self.pg_learning_rate = 0.01
        self.pg_baseline_decay = 0.9
        
        param_names = list(self.params.keys())
        n_params = len(param_names)
        
        # Policy network parameters (simple linear policy)
        self.key, subkey = random.split(self.key)
        self.pg_policy_params = {
            'mean': jnp.array([self.params[name] for name in param_names]),
            'log_std': jnp.zeros(n_params) - 1.0  # Initialize with std=exp(-1)≈0.37
        }
        
        # Baseline for variance reduction
        self.pg_baseline = 0.0
        self.pg_param_names = param_names
        
        # Episode history
        self.pg_episode_rewards = []
        self.pg_episode_log_probs = []
    
    def _setup_actor_critic(self):
        """Set up Actor-Critic for parameter optimization."""
        self.ac_actor_lr = 0.01
        self.ac_critic_lr = 0.02
        self.ac_gamma = 0.95
        
        param_names = list(self.params.keys())
        n_params = len(param_names)
        
        # Actor network (policy) parameters
        self.key, subkey1, subkey2 = random.split(self.key, 3)
        
        # Simple linear actor
        self.ac_actor_params = {
            'mean': jnp.array([self.params[name] for name in param_names]),
            'log_std': jnp.zeros(n_params) - 1.0
        }
        
        # Simple linear critic (value function)
        self.ac_critic_params = {
            'weights': random.normal(subkey2, (n_params, 1)) * 0.1,
            'bias': jnp.array([0.0])
        }
        
        self.ac_param_names = param_names
        
        # Experience buffer
        self.ac_states = []
        self.ac_actions = []
        self.ac_rewards = []
        self.ac_values = []
    
    def _setup_multi_agent_rl(self):
        """Set up Multi-Agent RL for parameter optimization."""
        # Each parameter is controlled by a separate agent
        self.marl_n_agents = len(self.params)
        self.marl_learning_rate = 0.01
        self.marl_epsilon = 0.1
        self.marl_epsilon_decay = 0.995
        
        param_names = list(self.params.keys())
        
        # Each agent has its own Q-table for its parameter
        self.marl_agents = {}
        for i, param in enumerate(param_names):
            min_val, max_val = self.param_bounds[param]
            
            # Discretize this parameter's space
            param_bins = jnp.linspace(min_val, max_val, 10)
            n_states = len(param_bins)
            n_actions = 3  # decrease, stay, increase
            
            self.marl_agents[param] = {
                'q_table': jnp.zeros((n_states, n_actions)),
                'param_bins': param_bins,
                'current_state': self._find_nearest_bin(self.params[param], param_bins),
                'epsilon': self.marl_epsilon,
                'alpha': 0.1,
                'gamma': 0.9
            }
        
        self.marl_param_names = param_names
        self.marl_step_size = 0.05
    
    def _setup_deep_q_network(self):
        """Set up Deep Q-Network (DQN) for parameter optimization."""
        self.dqn_learning_rate = 0.001
        self.dqn_epsilon = 0.1
        self.dqn_epsilon_decay = 0.995
        self.dqn_gamma = 0.95
        self.dqn_batch_size = 32
        self.dqn_memory_size = 1000
        
        param_names = list(self.params.keys())
        n_params = len(param_names)
        n_actions = 2 * n_params  # +/- for each parameter
        
        # Simple neural network for Q-function
        self.key, subkey1, subkey2 = random.split(self.key, 3)
        hidden_size = 64
        
        self.dqn_params = {
            'layer1': {
                'weights': random.normal(subkey1, (n_params, hidden_size)) * 0.1,
                'bias': jnp.zeros(hidden_size)
            },
            'layer2': {
                'weights': random.normal(subkey2, (hidden_size, n_actions)) * 0.1,
                'bias': jnp.zeros(n_actions)
            }
        }
        
        # Experience replay buffer
        self.dqn_memory = []
        self.dqn_param_names = param_names
        self.dqn_step_size = 0.05
    
    def _params_to_state(self, params: Dict[str, float]) -> Tuple[int, ...]:
        """Convert continuous parameters to discrete state for Q-learning."""
        state = []
        for param in self.ql_param_names:
            value = params[param]
            bins = self.ql_param_bins[param]
            # Find nearest bin
            bin_idx = jnp.argmin(jnp.abs(bins - value))
            state.append(int(bin_idx))
        return tuple(state)
    
    def _find_nearest_bin(self, value: float, bins: jax.Array) -> int:
        """Find the nearest bin index for a value."""
        return int(jnp.argmin(jnp.abs(bins - value)))
    
    def _dqn_forward(self, state: jax.Array) -> jax.Array:
        """Forward pass through DQN."""
        # Layer 1
        h1 = jnp.tanh(jnp.dot(state, self.dqn_params['layer1']['weights']) + 
                      self.dqn_params['layer1']['bias'])
        # Layer 2 (output)
        q_values = jnp.dot(h1, self.dqn_params['layer2']['weights']) + \
                   self.dqn_params['layer2']['bias']
        return q_values
    
    def _policy_gradient_sample_action(self, state: jax.Array) -> Tuple[jax.Array, float]:
        """Sample action from policy and return log probability."""
        mean = self.pg_policy_params['mean']
        std = jnp.exp(self.pg_policy_params['log_std'])
        
        # Sample from normal distribution
        self.key, subkey = random.split(self.key)
        action = mean + std * random.normal(subkey, mean.shape)
        
        # Compute log probability
        log_prob = -0.5 * jnp.sum(((action - mean) / std) ** 2) - \
                   0.5 * jnp.sum(jnp.log(2 * jnp.pi * std ** 2))
        
        return action, float(log_prob)
    
    def _actor_critic_forward(self, state: jax.Array) -> Tuple[jax.Array, float, float]:
        """Forward pass for actor-critic."""
        # Actor (policy)
        mean = self.ac_actor_params['mean']
        std = jnp.exp(self.ac_actor_params['log_std'])
        
        # Sample action
        self.key, subkey = random.split(self.key)
        action = mean + std * random.normal(subkey, mean.shape)
        
        # Log probability
        log_prob = -0.5 * jnp.sum(((action - mean) / std) ** 2) - \
                   0.5 * jnp.sum(jnp.log(2 * jnp.pi * std ** 2))
        
        # Critic (value function)
        value = jnp.dot(state, self.ac_critic_params['weights']).squeeze() + \
                self.ac_critic_params['bias'][0]
        
        return action, float(log_prob), float(value)
    
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
        """Q-Learning calibration."""
        no_improvement_count = 0
        
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\nIteration {iteration+1}/{self.max_iterations}")
            
            # Get current state
            current_state = self._params_to_state(self.params)
            
            # Epsilon-greedy action selection
            self.key, subkey = random.split(self.key)
            if random.uniform(subkey) < self.ql_epsilon:
                # Explore: random action
                action = random.randint(subkey, (), 0, self.ql_n_actions)
            else:
                # Exploit: best action
                q_values = self.ql_q_table[current_state]
                action = jnp.argmax(q_values)
            
            # Apply action to parameters
            new_params = self.params.copy()
            param_idx = int(action) // 2
            direction = 1 if int(action) % 2 == 0 else -1
            param_name = self.ql_param_names[param_idx]
            
            # Update parameter
            min_val, max_val = self.param_bounds[param_name]
            step = self.ql_step_size * (max_val - min_val)
            new_value = self.params[param_name] + direction * step
            new_params[param_name] = float(jnp.clip(new_value, min_val, max_val))
            
            # Evaluate new parameters
            loss, ci = self._evaluate_params_robust(new_params)
            reward = -loss  # Negative loss as reward
            
            # Get next state
            next_state = self._params_to_state(new_params)
            
            # Q-learning update
            current_q = self.ql_q_table[current_state + (int(action),)]
            next_max_q = jnp.max(self.ql_q_table[next_state])
            target_q = reward + self.ql_gamma * next_max_q
            
            # Update Q-table
            self.ql_q_table = self.ql_q_table.at[current_state + (int(action),)].set(
                current_q + self.ql_alpha * (target_q - current_q)
            )
            
            # Update parameters and tracking
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
                print(f"Epsilon: {self.ql_epsilon:.3f}")
            
            # Decay exploration
            self.ql_epsilon *= self.ql_epsilon_decay
            
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
    
    def _calibrate_policy_gradient(self, verbose: bool) -> Dict[str, float]:
        """Policy Gradient (REINFORCE) calibration."""
        no_improvement_count = 0
        
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\nIteration {iteration+1}/{self.max_iterations}")
            
            # Sample action from policy
            state = jnp.array([self.params[name] for name in self.pg_param_names])
            action, log_prob = self._policy_gradient_sample_action(state)
            
            # Convert action to parameter dictionary and clip to bounds
            new_params = {}
            for i, param_name in enumerate(self.pg_param_names):
                min_val, max_val = self.param_bounds[param_name]
                new_params[param_name] = float(jnp.clip(action[i], min_val, max_val))
            
            # Evaluate parameters
            loss, ci = self._evaluate_params_robust(new_params)
            reward = -loss  # Negative loss as reward
            
            # Store episode data
            self.pg_episode_rewards.append(reward)
            self.pg_episode_log_probs.append(log_prob)
            
            # Update baseline
            self.pg_baseline = self.pg_baseline_decay * self.pg_baseline + \
                              (1 - self.pg_baseline_decay) * reward
            
            # Policy gradient update
            advantage = reward - self.pg_baseline
            
            # Update policy parameters
            grad_log_prob_mean = (action - self.pg_policy_params['mean']) / \
                                jnp.exp(2 * self.pg_policy_params['log_std'])
            grad_log_prob_std = ((action - self.pg_policy_params['mean']) ** 2 / \
                                jnp.exp(2 * self.pg_policy_params['log_std']) - 1)
            
            # Apply gradients
            self.pg_policy_params['mean'] += self.pg_learning_rate * advantage * grad_log_prob_mean
            self.pg_policy_params['log_std'] += self.pg_learning_rate * advantage * grad_log_prob_std
            
            # Update current parameters to the mean of the policy
            for i, param_name in enumerate(self.pg_param_names):
                min_val, max_val = self.param_bounds[param_name]
                self.params[param_name] = float(jnp.clip(
                    self.pg_policy_params['mean'][i], min_val, max_val
                ))
            
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
                print(f"Policy mean: {self.pg_policy_params['mean']}")
            
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
    
    def _calibrate_actor_critic(self, verbose: bool) -> Dict[str, float]:
        """Actor-Critic calibration."""
        no_improvement_count = 0
        
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\nIteration {iteration+1}/{self.max_iterations}")
            
            # Current state
            state = jnp.array([self.params[name] for name in self.ac_param_names])
            
            # Forward pass
            action, log_prob, value = self._actor_critic_forward(state)
            
            # Convert action to parameter dictionary and clip to bounds
            new_params = {}
            for i, param_name in enumerate(self.ac_param_names):
                min_val, max_val = self.param_bounds[param_name]
                new_params[param_name] = float(jnp.clip(action[i], min_val, max_val))
            
            # Evaluate parameters
            loss, ci = self._evaluate_params_robust(new_params)
            reward = -loss  # Negative loss as reward
            
            # Next state value (for TD error)
            next_state = jnp.array([new_params[name] for name in self.ac_param_names])
            next_value = jnp.dot(next_state, self.ac_critic_params['weights']).squeeze() + \
                        self.ac_critic_params['bias'][0]
            
            # TD error
            td_target = reward + self.ac_gamma * next_value
            td_error = td_target - value
            
            # Update critic
            critic_grad_weights = jnp.outer(state, td_error)
            critic_grad_bias = jnp.array([td_error])
            
            self.ac_critic_params['weights'] += self.ac_critic_lr * critic_grad_weights
            self.ac_critic_params['bias'] += self.ac_critic_lr * critic_grad_bias
            
            # Update actor
            actor_grad_mean = td_error * (action - self.ac_actor_params['mean']) / \
                             jnp.exp(2 * self.ac_actor_params['log_std'])
            actor_grad_std = td_error * ((action - self.ac_actor_params['mean']) ** 2 / \
                            jnp.exp(2 * self.ac_actor_params['log_std']) - 1)
            
            self.ac_actor_params['mean'] += self.ac_actor_lr * actor_grad_mean
            self.ac_actor_params['log_std'] += self.ac_actor_lr * actor_grad_std
            
            # Update current parameters
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
                print(f"Reward: {reward:.6f}, TD Error: {td_error:.6f}")
                print(f"Value: {value:.6f}")
            
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
    
    def _calibrate_multi_agent_rl(self, verbose: bool) -> Dict[str, float]:
        """Multi-Agent RL calibration."""
        no_improvement_count = 0
        
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\nIteration {iteration+1}/{self.max_iterations}")
            
            # Each agent selects an action for its parameter
            new_params = self.params.copy()
            agent_actions = {}
            
            for param_name in self.marl_param_names:
                agent = self.marl_agents[param_name]
                current_state = agent['current_state']
                
                # Epsilon-greedy action selection
                self.key, subkey = random.split(self.key)
                if random.uniform(subkey) < agent['epsilon']:
                    action = random.randint(subkey, (), 0, 3)  # 0: decrease, 1: stay, 2: increase
                else:
                    q_values = agent['q_table'][current_state]
                    action = jnp.argmax(q_values)
                
                agent_actions[param_name] = int(action)
                
                # Apply action
                min_val, max_val = self.param_bounds[param_name]
                step = self.marl_step_size * (max_val - min_val)
                
                if action == 0:  # decrease
                    new_value = self.params[param_name] - step
                elif action == 1:  # stay
                    new_value = self.params[param_name]
                else:  # increase
                    new_value = self.params[param_name] + step
                
                new_params[param_name] = float(jnp.clip(new_value, min_val, max_val))
            
            # Evaluate joint action
            loss, ci = self._evaluate_params_robust(new_params)
            reward = -loss  # Negative loss as reward
            
            # Update each agent's Q-table
            for param_name in self.marl_param_names:
                agent = self.marl_agents[param_name]
                current_state = agent['current_state']
                action = agent_actions[param_name]
                
                # Find next state
                next_state = self._find_nearest_bin(new_params[param_name], agent['param_bins'])
                
                # Q-learning update
                current_q = agent['q_table'][current_state, action]
                next_max_q = jnp.max(agent['q_table'][next_state])
                target_q = reward + agent['gamma'] * next_max_q
                
                # Update Q-table
                agent['q_table'] = agent['q_table'].at[current_state, action].set(
                    current_q + agent['alpha'] * (target_q - current_q)
                )
                
                # Update agent's current state
                agent['current_state'] = next_state
                
                # Decay exploration
                agent['epsilon'] *= self.marl_epsilon_decay
            
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
    
    def _calibrate_dqn(self, verbose: bool) -> Dict[str, float]:
        """Deep Q-Network calibration."""
        no_improvement_count = 0
        
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\nIteration {iteration+1}/{self.max_iterations}")
            
            # Current state
            state = jnp.array([self.params[name] for name in self.dqn_param_names])
            
            # Epsilon-greedy action selection
            self.key, subkey = random.split(self.key)
            if random.uniform(subkey) < self.dqn_epsilon:
                # Explore: random action
                action = random.randint(subkey, (), 0, len(self.dqn_param_names) * 2)
            else:
                # Exploit: best action from DQN
                q_values = self._dqn_forward(state)
                action = jnp.argmax(q_values)
            
            # Apply action to parameters
            new_params = self.params.copy()
            param_idx = int(action) // 2
            direction = 1 if int(action) % 2 == 0 else -1
            param_name = self.dqn_param_names[param_idx]
            
            # Update parameter
            min_val, max_val = self.param_bounds[param_name]
            step = self.dqn_step_size * (max_val - min_val)
            new_value = self.params[param_name] + direction * step
            new_params[param_name] = float(jnp.clip(new_value, min_val, max_val))
            
            # Evaluate new parameters
            loss, ci = self._evaluate_params_robust(new_params)
            reward = -loss  # Negative loss as reward
            
            # Store experience
            next_state = jnp.array([new_params[name] for name in self.dqn_param_names])
            experience = (state, int(action), reward, next_state, False)  # not done
            
            self.dqn_memory.append(experience)
            if len(self.dqn_memory) > self.dqn_memory_size:
                self.dqn_memory.pop(0)
            
            # Train DQN if we have enough experiences
            if len(self.dqn_memory) >= self.dqn_batch_size:
                self._train_dqn()
            
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
            
            # Decay exploration
            self.dqn_epsilon *= self.dqn_epsilon_decay
            
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
    
    def _train_dqn(self):
        """Train the DQN using experience replay."""
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
        
        # Compute target Q-values
        next_q_values = jnp.array([self._dqn_forward(next_state) for next_state in next_states])
        target_q_values = rewards + self.dqn_gamma * jnp.max(next_q_values, axis=1)
        
        # Compute current Q-values
        current_q_values = jnp.array([self._dqn_forward(state) for state in states])
        
        # Compute loss and gradients (simplified gradient descent)
        for i in range(len(batch)):
            state = states[i]
            action = actions[i]
            target = target_q_values[i]
            current = current_q_values[i, action]
            
            # Simple gradient update (this is a simplified version)
            error = target - current
            
            # Update network parameters (simplified)
            # In practice, you'd use proper backpropagation
            h1 = jnp.tanh(jnp.dot(state, self.dqn_params['layer1']['weights']) + 
                         self.dqn_params['layer1']['bias'])
            
            # Gradient for layer 2
            grad_w2 = error * h1
            grad_b2 = error
            
            # Update layer 2
            self.dqn_params['layer2']['weights'] = self.dqn_params['layer2']['weights'].at[action].add(
                self.dqn_learning_rate * grad_w2
            )
            self.dqn_params['layer2']['bias'] = self.dqn_params['layer2']['bias'].at[action].add(
                self.dqn_learning_rate * grad_b2
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
    max_iterations: int = 50,
    verbose: bool = True
) -> Dict[str, Any]:
    """Compare different calibration methods on the same problem.
    
    Args:
        model_factory: Function to create model instances
        initial_params: Initial parameter values
        target_metrics: Target metric values
        methods: List of methods to compare
        max_iterations: Maximum iterations for each method
        verbose: Whether to print progress
        
    Returns:
        Dictionary with comparison results
    """
    ensemble = EnsembleCalibrator(
        model_factory=model_factory,
        initial_params=initial_params,
        target_metrics=target_metrics,
        methods=methods,
        max_iterations=max_iterations
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