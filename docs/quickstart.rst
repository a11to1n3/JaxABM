Quick Start Guide
=================

This guide will get you up and running with JaxABM in just a few minutes.

Basic Concepts
--------------

JaxABM is built around these core concepts:

- **Model**: The main simulation container
- **Agents**: Individual entities with behaviors
- **AgentCollection**: Groups of similar agents
- **Environment**: Shared state and resources
- **Calibration**: Parameter optimization
- **Sensitivity Analysis**: Parameter importance testing

Your First Model
----------------

Let's create a simple random walk model:

.. code-block:: python

   import jaxabm as jx
   import jax.numpy as jnp
   import jax.random as random

   # Create model
   model = jx.Model()

   # Define agent behavior
   def step_function(agents, env_state):
       """Agents take random steps"""
       key = random.PRNGKey(42)
       
       # Random movement
       dx = random.normal(key, (agents.n_agents,)) * 0.1
       dy = random.normal(key, (agents.n_agents,)) * 0.1
       
       # Update positions
       new_x = agents.state['x'] + dx
       new_y = agents.state['y'] + dy
       
       return agents.update_state({'x': new_x, 'y': new_y})

   # Create agents
   agents = jx.AgentCollection(
       "walkers", 
       n_agents=100,
       initial_state={
           'x': jnp.zeros(100),
           'y': jnp.zeros(100)
       },
       step_function=step_function
   )

   # Add agents to model
   model.add_agent_collection(agents)

   # Run simulation
   results = model.run(steps=50)
   
   print(f"Final positions: {results.agents['walkers'].state}")

Parameter Calibration
--------------------

JaxABM provides powerful calibration tools to find optimal parameters:

.. code-block:: python

   import jaxabm as jx

   # Define model factory
   def create_model(params):
       class EconomyModel:
           def __init__(self, params):
               self.growth_rate = params.get('growth_rate', 0.02)
               self.volatility = params.get('volatility', 0.1)
           
           def run(self, steps=100):
               # Simulate economic growth
               growth = self.growth_rate + random.normal(key, ()) * self.volatility
               return {'gdp': [growth], 'unemployment': [0.05]}
       
       return EconomyModel(params)

   # Set up calibration
   calibrator = jx.analysis.ModelCalibrator(
       model_factory=create_model,
       initial_params={'growth_rate': 0.01, 'volatility': 0.05},
       target_metrics={'gdp': 0.025, 'unemployment': 0.04},
       param_bounds={'growth_rate': (0.0, 0.1), 'volatility': (0.01, 0.5)},
       method='pso'  # Particle Swarm Optimization
   )

   # Run calibration
   best_params = calibrator.calibrate()
   print(f"Optimal parameters: {best_params}")

Reinforcement Learning Calibration
----------------------------------

Use RL methods for complex parameter optimization:

.. code-block:: python

   # Use Actor-Critic for advanced calibration
   rl_calibrator = jx.analysis.ModelCalibrator(
       model_factory=create_model,
       initial_params={'growth_rate': 0.01, 'volatility': 0.05},
       target_metrics={'gdp': 0.025, 'unemployment': 0.04},
       param_bounds={'growth_rate': (0.0, 0.1), 'volatility': (0.01, 0.5)},
       method='actor_critic',  # RL method
       max_iterations=50
   )

   rl_params = rl_calibrator.calibrate()

Available RL methods:
- ``q_learning``: Q-learning with neural networks
- ``policy_gradient``: Policy gradient methods
- ``actor_critic``: Actor-critic algorithms  
- ``dqn``: Deep Q-Networks

Sensitivity Analysis
-------------------

Analyze parameter importance:

.. code-block:: python

   # Set up sensitivity analysis
   sensitivity = jx.analysis.SensitivityAnalysis(
       model_factory=create_model,
       param_ranges={
           'growth_rate': (0.0, 0.1),
           'volatility': (0.01, 0.5)
       },
       metrics_of_interest=['gdp', 'unemployment']
   )

   # Run analysis
   results = sensitivity.run()

   # Get Sobol indices
   indices = sensitivity.sobol_indices()
   print(f"Parameter importance: {indices}")

   # Plot results
   sensitivity.plot()

Advanced Features
-----------------

Multi-Agent Interactions
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def interaction_function(agents, env_state):
       """Agents interact with each other"""
       # Find neighbors
       distances = compute_distances(agents.state['x'], agents.state['y'])
       neighbors = find_neighbors(distances, radius=0.5)
       
       # Influence behavior
       influence = compute_social_influence(agents.state, neighbors)
       
       return agents.update_state({'influence': influence})

Custom Environment
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Add environment state
   model.add_env_state('temperature', initial_value=20.0)
   model.add_env_state('resources', initial_value=jnp.ones(100))

   # Environment update function
   def update_environment(env_state, agents):
       # Climate change
       new_temp = env_state['temperature'] + 0.01
       
       # Resource depletion
       consumption = jnp.sum(agents['consumers'].state['consumption'])
       new_resources = env_state['resources'] - consumption
       
       return {'temperature': new_temp, 'resources': new_resources}

   model.set_env_step_function(update_environment)

Performance Optimization
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from jax import jit, vmap

   # JIT compile for speed
   @jit
   def fast_step_function(agents, env_state):
       # Your agent logic here
       return agents

   # Vectorize operations
   @vmap
   def vectorized_computation(agent_data):
       # Process each agent
       return result

Next Steps
----------

Now that you have the basics, explore:

1. **Detailed Tutorials**: :doc:`tutorials/index`
2. **Complete Examples**: :doc:`examples/index` 
3. **Calibration Guide**: :doc:`calibration/index`
4. **API Reference**: :doc:`api/index`

Common Patterns
---------------

**Economic Models**
   See :doc:`examples/economic_model` for a complete economic simulation

**Epidemiological Models**
   Check :doc:`examples/sir_model` for disease spread modeling

**Social Network Models**
   Explore :doc:`examples/social_network` for network-based simulations

**Spatial Models**
   Look at :doc:`examples/spatial_model` for geographic simulations 