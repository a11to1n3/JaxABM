Contributing to JaxABM
=====================

We welcome contributions to JaxABM! This guide will help you get started with contributing to the project.

Ways to Contribute
------------------

There are many ways to contribute to JaxABM:

- **Report bugs** and suggest features
- **Improve documentation** and examples
- **Add new calibration methods** or optimization algorithms
- **Create example models** for different domains
- **Optimize performance** and scalability
- **Fix bugs** and improve code quality

Getting Started
---------------

Development Setup
^^^^^^^^^^^^^^^^^

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:

   .. code-block:: bash

      git clone https://github.com/yourusername/jaxabm.git
      cd jaxabm

3. **Create a virtual environment**:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

4. **Install development dependencies**:

   .. code-block:: bash

      pip install -e ".[dev]"

5. **Run tests** to ensure everything works:

   .. code-block:: bash

      pytest tests/

Development Workflow
^^^^^^^^^^^^^^^^^^^^

1. **Create a feature branch**:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

2. **Make your changes** with good commit messages
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Run tests and checks** before submitting
6. **Submit a pull request** with a clear description

Code Standards
--------------

Code Style
^^^^^^^^^^

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black formatter)
- **Imports**: Use absolute imports when possible
- **Docstrings**: Google style docstrings
- **Type hints**: Use type hints for all public functions

Example:

.. code-block:: python

   def calibrate_model(
       model_factory: ModelFactory,
       initial_params: Dict[str, float],
       target_metrics: Dict[str, float],
       method: str = "actor_critic"
   ) -> Dict[str, float]:
       """Calibrate model parameters to match target metrics.
       
       Args:
           model_factory: Function that creates model instances
           initial_params: Starting parameter values
           target_metrics: Desired metric outcomes
           method: Calibration method to use
           
       Returns:
           Dictionary of optimal parameter values
           
       Raises:
           ValueError: If method is not supported
       """
       # Implementation here
       pass

Testing Requirements
^^^^^^^^^^^^^^^^^^^^

All contributions must include appropriate tests:

**Unit Tests**
   - Test individual functions and classes
   - Use ``pytest`` framework
   - Place in ``tests/unit/`` directory
   - Cover edge cases and error conditions

**Integration Tests**
   - Test complete workflows
   - Place in ``tests/integration/`` directory
   - Include realistic use cases

**Example Test**:

.. code-block:: python

   import pytest
   import jaxabm as jx

   def test_model_calibrator_initialization():
       """Test ModelCalibrator can be initialized correctly."""
       def dummy_factory(params):
           return DummyModel(params)
       
       calibrator = jx.analysis.ModelCalibrator(
           model_factory=dummy_factory,
           initial_params={'param1': 0.5},
           target_metrics={'metric1': 1.0}
       )
       
       assert calibrator.method == "adam"  # default
       assert calibrator.params['param1'] == 0.5

Documentation Standards
^^^^^^^^^^^^^^^^^^^^^^^

**Docstrings**
   - Use Google style docstrings
   - Include parameter types and descriptions
   - Provide usage examples for complex functions
   - Document all public APIs

**Code Comments**
   - Explain complex algorithms and mathematical operations
   - Use comments to clarify non-obvious design decisions
   - Keep comments up-to-date with code changes

**Documentation Files**
   - Use reStructuredText (RST) format
   - Include code examples that work
   - Cross-reference related documentation
   - Update API documentation when adding features

Contribution Areas
------------------

High-Priority Areas
^^^^^^^^^^^^^^^^^^^

1. **More Calibration Methods**
   - Bayesian optimization improvements
   - Multi-objective optimization algorithms
   - Distributed/parallel calibration methods

2. **Performance Optimization**
   - Better GPU utilization
   - Memory usage optimization
   - Faster convergence algorithms

3. **Example Models**
   - Domain-specific model templates
   - Real-world case studies
   - Educational examples

4. **Documentation**
   - More detailed tutorials
   - Performance benchmarking guides
   - Best practices documentation

New Calibration Methods
^^^^^^^^^^^^^^^^^^^^^^^

When adding new calibration methods:

1. **Inherit from base classes** or follow established patterns
2. **Add comprehensive tests** including edge cases
3. **Document the method** with references to papers
4. **Provide usage examples** showing when to use the method
5. **Benchmark performance** against existing methods

Example structure:

.. code-block:: python

   def _setup_new_method(self):
       """Set up new calibration method."""
       # Initialize method-specific parameters
       pass
   
   def _calibrate_new_method(self, verbose: bool) -> Dict[str, float]:
       """Run new calibration method."""
       # Implementation
       return best_params

Example Models
^^^^^^^^^^^^^^

Good example models should:

- **Address real-world problems** or educational concepts
- **Include clear documentation** explaining the model
- **Provide realistic parameters** and expected outcomes
- **Include visualization** of results
- **Demonstrate calibration** when appropriate

Review Process
--------------

Pull Request Guidelines
^^^^^^^^^^^^^^^^^^^^^^^

When submitting a pull request:

1. **Provide clear description** of changes and motivation
2. **Include tests** for new functionality
3. **Update documentation** as needed
4. **Ensure all tests pass** on all supported platforms
5. **Respond to review feedback** promptly

Review Criteria
^^^^^^^^^^^^^^^

We review contributions based on:

- **Code quality** and adherence to standards
- **Test coverage** and quality
- **Documentation** completeness and clarity
- **Performance impact** on existing functionality
- **API consistency** with existing patterns

Getting Help
------------

If you need help with contributing:

1. **Check existing issues** on GitHub
2. **Ask questions** in discussions or issues
3. **Review existing code** for patterns and examples
4. **Start with small contributions** to get familiar

Communication
^^^^^^^^^^^^^

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Pull Requests**: Code contributions and reviews

Code of Conduct
---------------

We are committed to providing a welcoming and inclusive environment. Please:

- **Be respectful** of different viewpoints and experiences
- **Provide constructive feedback** during reviews
- **Help others** learn and contribute
- **Follow professional standards** in all interactions

Recognition
-----------

Contributors are recognized in:

- **Changelog** for significant contributions
- **Documentation** credits
- **GitHub contributors** list
- **Release notes** for major features

Thank you for contributing to JaxABM! Your efforts help make agent-based modeling more accessible and powerful for researchers and practitioners worldwide. 