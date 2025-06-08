Examples
========

Complete, runnable examples demonstrating JaxABM capabilities.

Featured Examples
-----------------

.. toctree::
   :maxdepth: 1

   economic_model
   sir_model
   social_network
   spatial_model
   financial_markets

Model Types
-----------

Economic Models
^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   economic/macro_economy
   economic/market_dynamics
   economic/consumer_behavior
   economic/supply_chain

Epidemiological Models
^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   epidemiology/sir_basic
   epidemiology/seir_model
   epidemiology/network_spread
   epidemiology/vaccination_strategies

Social Models
^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   social/opinion_dynamics
   social/social_networks
   social/cultural_evolution
   social/collective_behavior

Spatial Models
^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   spatial/predator_prey
   spatial/urban_growth
   spatial/migration_patterns
   spatial/resource_competition

Calibration Examples
--------------------

These examples focus on parameter calibration and optimization:

.. toctree::
   :maxdepth: 1

   calibration/rl_economic_calibration
   calibration/multi_objective_optimization
   calibration/sensitivity_driven_calibration
   calibration/ensemble_methods

Running Examples
----------------

All examples can be run directly:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/username/jaxabm.git
   cd jaxabm

   # Run any example
   python examples/models/predator_prey.py

Or explore them in Jupyter notebooks:

.. code-block:: bash

   jupyter notebook examples/notebooks/

Example Categories
------------------

**Basic Examples** (⭐)
   Simple models perfect for learning JaxABM basics.

**Intermediate Examples** (⭐⭐)
   More complex models with multiple agent types and interactions.

**Advanced Examples** (⭐⭐⭐)
   Sophisticated models showcasing advanced features like RL calibration.

Quick Reference
---------------

================================  ============  ===============  ================
Example                          Complexity    Domain           Key Features
================================  ============  ===============  ================
Random Walk                      ⭐             General          Basic movement
Predator-Prey                    ⭐⭐           Ecology          Spatial dynamics
Economic Growth                  ⭐⭐           Economics        Parameter calibration
SIR Epidemic                     ⭐⭐           Epidemiology     Network spread
Opinion Dynamics                 ⭐⭐⭐         Social           Complex interactions
Financial Markets                ⭐⭐⭐         Finance          RL optimization
================================  ============  ===============  ================

Code Structure
--------------

Each example follows this structure:

.. code-block:: python

   import jaxabm as jx
   import jax.numpy as jnp

   # 1. Model Definition
   class ExampleModel:
       def __init__(self, params):
           # Initialize model parameters
           pass
       
       def run(self, steps):
           # Run simulation
           pass

   # 2. Agent Behaviors
   def agent_step_function(agents, env_state):
       # Define agent behavior
       return updated_agents

   # 3. Calibration (if applicable)
   def setup_calibration():
       calibrator = jx.analysis.ModelCalibrator(...)
       return calibrator.calibrate()

   # 4. Analysis and Visualization
   def analyze_results(results):
       # Plot and analyze outcomes
       pass

   if __name__ == "__main__":
       # Run the example
       main()

Getting Started
---------------

1. **Choose an example** that matches your interest/domain
2. **Read the documentation** to understand the model
3. **Run the code** to see it in action
4. **Modify parameters** to explore behavior
5. **Extend the model** with your own features

Each example includes:

- **Model description** and motivation
- **Complete source code** with comments
- **Parameter explanations** and sensible defaults
- **Visualization code** for results
- **Extension suggestions** for further exploration

Contributing Examples
---------------------

We welcome new examples! Please see our contribution guidelines for:

- Code style requirements
- Documentation standards
- Testing expectations
- Review process

Good examples include:

- Clear, well-commented code
- Realistic parameter values
- Meaningful visualizations
- Educational value
- Proper citations for published models 