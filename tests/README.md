# JaxABM Test Suite

This directory contains the complete test suite for the JaxABM framework, including both unit tests and integration tests.

## Test Structure

The test suite is organized as follows:

- `unit/`: Unit tests for individual components
  - `test_agent.py`: Tests for AgentType, AgentCollection, and ModelConfig
  - `test_model.py`: Tests for the Model class
  - `test_analysis.py`: Tests for SensitivityAnalysis and ModelCalibrator

- `integration/`: Integration tests for complete workflows
  - `test_integration.py`: Tests for complete modeling workflows

- `conftest.py`: Pytest fixtures and configuration
- `run_tests.py`: Test runner script

## Running Tests

You can run tests using either the provided test runner script or directly with pytest.

### Using the Test Runner

The test runner script supports both unittest and pytest:

```bash
# Run all tests
./tests/run_tests.py

# Run with verbose output
./tests/run_tests.py -v

# Run only unittest-based tests
./tests/run_tests.py --unittest-only

# Run only pytest-based tests
./tests/run_tests.py --pytest-only

# Run tests matching a specific pattern
./tests/run_tests.py -p "*agent*"
```

### Using Pytest Directly

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/unit/test_agent.py

# Run tests matching a specific pattern
pytest -k "agent"

# Run with coverage report
pytest --cov=jaxabm

# Run tests in parallel
pytest -xvs -n auto
```

### Using Unittest Directly

```bash
# Run all tests
python -m unittest discover

# Run a specific test file
python -m unittest tests/unit/test_agent.py

# Run a specific test case
python -m unittest tests.unit.test_agent.TestAgentCollection
```

## Writing Tests

When writing new tests:

1. **Unit Tests**: Place these in the `unit/` directory and focus on testing a single component in isolation.
2. **Integration Tests**: Place these in the `integration/` directory and test interactions between multiple components.

All test files should follow the naming convention `test_*.py`.

### Using Fixtures

Common test fixtures are available in `conftest.py`. These include:

- `random_seed`: A fixed random seed (42) for deterministic tests
- `random_key`: A fixed JAX PRNG key based on the random seed
- `simple_agent_type`: A simple agent type implementation for testing
- `simple_agent_collection`: A pre-initialized agent collection
- `simple_model`: A pre-initialized model

Example usage with pytest:

```python
def test_something(simple_model):
    # Use the model fixture
    results = simple_model.run()
    assert 'counter' in results
``` 