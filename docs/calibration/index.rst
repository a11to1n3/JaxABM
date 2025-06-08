Model Calibration
=================

JaxABM provides state-of-the-art parameter calibration methods to find optimal model parameters that match target outcomes. The calibration system supports traditional optimization methods as well as cutting-edge reinforcement learning approaches.

Overview
--------

Parameter calibration is the process of finding model parameters that produce outputs matching desired target metrics. JaxABM's calibration system is designed for:

- **Flexibility**: Multiple optimization algorithms
- **Performance**: JAX-accelerated computations
- **Robustness**: Handles complex, non-linear parameter spaces
- **Advanced Methods**: Reinforcement learning for challenging problems

Quick Example
-------------

.. code-block:: python

   import jaxabm as jx

   # Create calibrator
   calibrator = jx.analysis.ModelCalibrator(
       model_factory=your_model_factory,
       initial_params={'param1': 0.1, 'param2': 0.5},
       target_metrics={'output1': 2.5, 'output2': 1.0},
       method='actor_critic'  # RL method
   )

   # Find optimal parameters
   best_params = calibrator.calibrate()

Available Methods
-----------------

Traditional Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 30 25 25

   * - Method
     - Description
     - Best For
     - Performance
   * - ``adam``
     - Gradient-based optimization
     - Smooth landscapes
     - Fast
   * - ``pso``
     - Particle Swarm Optimization
     - Multi-modal problems
     - Good
   * - ``es``
     - Evolution Strategies
     - Noisy functions
     - Robust
   * - ``cem``
     - Cross-Entropy Method
     - High-dimensional spaces
     - Excellent

Reinforcement Learning Methods ⭐
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 30 25 25

   * - Method
     - Description
     - Best For
     - Performance
   * - ``actor_critic``
     - Actor-Critic algorithms
     - Complex dynamics
     - **Excellent**
   * - ``policy_gradient``
     - Policy gradient methods
     - Continuous spaces
     - Very Good
   * - ``q_learning``
     - Deep Q-Learning
     - Discrete actions
     - Good
   * - ``dqn``
     - Deep Q-Networks
     - Large state spaces
     - Very Good

.. note::
   **New in v0.1.0**: All RL methods have been significantly improved with enhanced numerical stability and convergence guarantees.

Method Selection Guide
----------------------

Choose your calibration method based on your problem characteristics:

**For Economic Models**: ``actor_critic`` or ``cem``
   - Handle complex parameter interactions
   - Robust to noisy evaluations
   - Good convergence properties

**For Epidemiological Models**: ``pso`` or ``policy_gradient``
   - Navigate multi-modal fitness landscapes
   - Handle discrete parameter changes well

**For Social Network Models**: ``es`` or ``dqn``
   - Robust to evaluation noise
   - Handle large parameter spaces

**For Quick Prototyping**: ``adam`` or ``pso``
   - Fast convergence for testing
   - Good general-purpose methods

Advanced Features
-----------------

Multi-Objective Calibration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   calibrator = jx.analysis.ModelCalibrator(
       model_factory=model_factory,
       initial_params=params,
       target_metrics={
           'accuracy': 0.95,
           'efficiency': 0.8,
           'robustness': 0.9
       },
       metrics_weights={
           'accuracy': 2.0,     # Higher priority
           'efficiency': 1.0,
           'robustness': 1.5
       },
       method='actor_critic'
   )

Parameter Bounds and Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   calibrator = jx.analysis.ModelCalibrator(
       model_factory=model_factory,
       initial_params={'growth': 0.02, 'volatility': 0.1},
       target_metrics={'gdp': 2.5, 'unemployment': 5.0},
       param_bounds={
           'growth': (0.0, 0.1),        # Economic constraints
           'volatility': (0.01, 0.5)    # Realistic ranges
       },
       method='policy_gradient'
   )

Robust Evaluation
^^^^^^^^^^^^^^^^^

.. code-block:: python

   calibrator = jx.analysis.ModelCalibrator(
       model_factory=model_factory,
       initial_params=params,
       target_metrics=targets,
       num_evaluation_runs=5,    # Multiple runs for robustness
       evaluation_steps=100,     # Longer simulations
       method='actor_critic'
   )

Detailed Guides
---------------

.. toctree::
   :maxdepth: 2

   traditional_methods
   reinforcement_learning
   advanced_techniques
   performance_tuning
   troubleshooting

Examples
--------

.. toctree::
   :maxdepth: 1

   examples/economic_calibration
   examples/epidemiological_calibration
   examples/social_network_calibration
   examples/multi_objective_calibration

Best Practices
--------------

1. **Start Simple**: Begin with traditional methods before using RL
2. **Parameter Scaling**: Normalize parameters to similar ranges
3. **Target Scaling**: Use relative errors for different metric scales
4. **Evaluation Budget**: Balance accuracy vs computational cost
5. **Convergence Monitoring**: Use early stopping and convergence criteria
6. **Reproducibility**: Set random seeds for consistent results

Performance Tips
----------------

- Use ``jit`` compilation for model functions
- Batch multiple evaluations when possible
- Consider GPU acceleration for large models
- Monitor memory usage with large agent populations
- Use appropriate tolerance levels for convergence 