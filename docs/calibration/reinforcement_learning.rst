Reinforcement Learning Calibration
==================================

JaxABM provides state-of-the-art reinforcement learning methods for parameter calibration. These methods are particularly effective for complex, non-linear parameter spaces where traditional optimization methods struggle.

Why Use RL for Calibration?
---------------------------

Reinforcement Learning approaches calibration as a sequential decision-making problem:

- **Adaptive Exploration**: Learns where to search in parameter space
- **Non-linear Dynamics**: Handles complex parameter interactions
- **Robust to Noise**: Naturally handles stochastic model outputs
- **Memory**: Learns from previous parameter evaluations
- **Continuous Learning**: Improves strategy over time

Available RL Methods
--------------------

Actor-Critic (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most robust and performant RL method for calibration.

.. code-block:: python

   calibrator = jx.analysis.ModelCalibrator(
       model_factory=model_factory,
       initial_params=initial_params,
       target_metrics=target_metrics,
       method='actor_critic',
       max_iterations=50
   )

**Strengths:**
- Excellent convergence properties
- Handles continuous parameter spaces naturally
- Low variance gradient estimates
- Fast learning

**Best For:**
- Economic models with complex interactions
- High-dimensional parameter spaces
- Models with smooth parameter landscapes

Policy Gradient
^^^^^^^^^^^^^^^

Direct policy optimization with enhanced numerical stability.

.. code-block:: python

   calibrator = jx.analysis.ModelCalibrator(
       model_factory=model_factory,
       initial_params=initial_params,
       target_metrics=target_metrics,
       method='policy_gradient',
       learning_rate=0.01
   )

**Strengths:**
- Direct optimization of parameter policy
- Good for continuous actions
- Principled exploration via entropy regularization

**Recent Improvements (v0.1.0):**
- Fixed numerical instability issues
- Added gradient clipping and value bounds
- Enhanced safety checks for NaN handling
- Improved convergence criteria

Q-Learning
^^^^^^^^^^

Neural network-based Q-learning for discrete parameter optimization.

.. code-block:: python

   calibrator = jx.analysis.ModelCalibrator(
       model_factory=model_factory,
       initial_params=initial_params,
       target_metrics=target_metrics,
       method='q_learning',
       max_iterations=100
   )

**Strengths:**
- Well-established theoretical foundations
- Good for discrete parameter choices
- Epsilon-greedy exploration

**Best For:**
- Models with discrete parameter choices
- When you need interpretable action-value functions

Deep Q-Networks (DQN)
^^^^^^^^^^^^^^^^^^^^^

Advanced Q-learning with experience replay and target networks.

.. code-block:: python

   calibrator = jx.analysis.ModelCalibrator(
       model_factory=model_factory,
       initial_params=initial_params,
       target_metrics=target_metrics,
       method='dqn',
       max_iterations=100
   )

**Strengths:**
- Stable learning with experience replay
- Handles large state spaces
- Double Q-learning for reduced overestimation

**Best For:**
- Complex models with large parameter spaces
- When sample efficiency is important

Configuration Options
---------------------

Learning Rate Tuning
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Conservative learning (stable but slow)
   calibrator = jx.analysis.ModelCalibrator(
       method='actor_critic',
       learning_rate=0.001
   )

   # Aggressive learning (fast but may be unstable)
   calibrator = jx.analysis.ModelCalibrator(
       method='actor_critic',
       learning_rate=0.01
   )

Exploration Control
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # High exploration (for complex landscapes)
   calibrator = jx.analysis.ModelCalibrator(
       method='policy_gradient',
       # More exploration via entropy regularization
   )

   # Conservative exploration (for smooth landscapes)
   calibrator = jx.analysis.ModelCalibrator(
       method='actor_critic',
       # Less exploration, more exploitation
   )

Advanced Techniques
-------------------

Custom Reward Shaping
^^^^^^^^^^^^^^^^^^^^^^

The RL methods use sophisticated reward shaping:

.. code-block:: python

   # Automatic reward shaping based on:
   # 1. Improvement over previous iteration
   # 2. Distance to target metrics
   # 3. Parameter bound violations
   # 4. Convergence bonuses

Multi-Objective RL
^^^^^^^^^^^^^^^^^^

Handle multiple competing objectives:

.. code-block:: python

   calibrator = jx.analysis.ModelCalibrator(
       target_metrics={
           'accuracy': 0.95,
           'speed': 100.0,
           'robustness': 0.9
       },
       metrics_weights={
           'accuracy': 2.0,      # Prioritize accuracy
           'speed': 1.0,
           'robustness': 1.5
       },
       method='actor_critic'
   )

Convergence Monitoring
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   calibrator = jx.analysis.ModelCalibrator(
       method='actor_critic',
       tolerance=1e-4,           # Convergence threshold
       patience=10,              # Early stopping patience
       max_iterations=100
   )

   # Monitor progress
   best_params = calibrator.calibrate(verbose=True)
   
   # Access training history
   history = calibrator.get_calibration_history()
   print(f"Loss progression: {history['loss']}")

Performance Optimization
------------------------

Batch Evaluation
^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Evaluate multiple parameter sets simultaneously
   calibrator = jx.analysis.ModelCalibrator(
       method='actor_critic',
       num_evaluation_runs=3,    # Average over multiple runs
       evaluation_steps=50       # Longer simulations for stability
   )

GPU Acceleration
^^^^^^^^^^^^^^^^

.. code-block:: python

   # RL methods automatically use GPU when available
   import jax
   print(f"Using devices: {jax.devices()}")

   # Ensure your model factory also uses GPU
   def gpu_model_factory(params):
       # Your model implementation
       return model

Memory Management
^^^^^^^^^^^^^^^^^

.. code-block:: python

   # For large models, control memory usage
   calibrator = jx.analysis.ModelCalibrator(
       method='dqn',
       max_iterations=50,        # Shorter runs
       evaluation_steps=25       # Shorter evaluations
   )

Troubleshooting
---------------

Common Issues and Solutions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Slow Convergence**
   - Increase learning rate
   - Use Actor-Critic instead of Policy Gradient
   - Check parameter bounds are reasonable

**Unstable Training**
   - Decrease learning rate
   - Increase evaluation_steps for more stable estimates
   - Use multiple evaluation runs

**Poor Final Performance**
   - Increase max_iterations
   - Check target metrics are achievable
   - Verify model factory is working correctly

**NaN Values (Fixed in v0.1.0)**
   - All RL methods now include robust NaN handling
   - Automatic gradient clipping prevents explosions
   - Safe numerical operations throughout

Debugging Tools
^^^^^^^^^^^^^^^

.. code-block:: python

   # Enable verbose output
   best_params = calibrator.calibrate(verbose=True)

   # Plot training progress
   calibrator.plot_calibration()

   # Examine parameter evolution
   history = calibrator.get_calibration_history()
   import matplotlib.pyplot as plt
   
   plt.figure(figsize=(12, 4))
   plt.subplot(1, 2, 1)
   plt.plot(history['loss'])
   plt.title('Loss Over Time')
   
   plt.subplot(1, 2, 2)
   for param_name in history['params'][0].keys():
       values = [p[param_name] for p in history['params']]
       plt.plot(values, label=param_name)
   plt.legend()
   plt.title('Parameter Evolution')
   plt.show()

Best Practices
--------------

1. **Start with Actor-Critic**: Most robust method for new problems
2. **Tune Learning Rates**: Start with 0.001-0.01 range
3. **Monitor Convergence**: Use verbose mode and check loss plots
4. **Parameter Scaling**: Normalize parameters to [0, 1] range when possible
5. **Evaluation Budget**: Balance between accuracy and computational cost
6. **Reproducibility**: Always set random seeds
7. **Validation**: Test final parameters on independent datasets

Example: Economic Model Calibration
-----------------------------------

Complete example using Actor-Critic for economic model:

.. code-block:: python

   import jaxabm as jx
   import jax.numpy as jnp

   def economic_model_factory(params):
       class EconomicModel:
           def __init__(self, params):
               self.growth_rate = params['growth_rate']
               self.inflation_rate = params['inflation_rate']
               self.employment_rate = params['employment_rate']
           
           def run(self, steps=100):
               # Complex economic dynamics
               gdp = 1.0
               for _ in range(steps):
                   gdp *= (1 + self.growth_rate)
                   # Add noise and interactions
                   gdp += jnp.random.normal() * 0.01
               
               # Compute metrics
               unemployment = max(0, 0.1 - self.employment_rate)
               inflation = self.inflation_rate * gdp
               
               return {
                   'gdp_growth': [gdp - 1.0],
                   'unemployment': [unemployment],
                   'inflation': [inflation]
               }
       
       return EconomicModel(params)

   # Set up RL calibration
   rl_calibrator = jx.analysis.ModelCalibrator(
       model_factory=economic_model_factory,
       initial_params={
           'growth_rate': 0.02,
           'inflation_rate': 0.03,
           'employment_rate': 0.95
       },
       target_metrics={
           'gdp_growth': 0.025,    # 2.5% growth
           'unemployment': 0.05,   # 5% unemployment
           'inflation': 0.02       # 2% inflation
       },
       param_bounds={
           'growth_rate': (0.0, 0.1),
           'inflation_rate': (0.0, 0.1),
           'employment_rate': (0.8, 1.0)
       },
       method='actor_critic',
       max_iterations=50,
       tolerance=1e-3
   )

   # Run calibration
   optimal_params = rl_calibrator.calibrate(verbose=True)
   print(f"Optimal parameters: {optimal_params}")

   # Validate results
   test_model = economic_model_factory(optimal_params)
   results = test_model.run()
   print(f"Final metrics: {results}")

This example demonstrates the power of RL methods for complex economic model calibration with multiple interacting parameters. 