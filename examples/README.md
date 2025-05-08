# JaxABM Example Scripts

This directory contains example scripts demonstrating various features of the JaxABM framework.

## Quick Start

Make sure you have activated your virtual environment and have all dependencies installed:

```bash
# From the project root
source .venv/bin/activate
pip install -e .
```

## Available Examples

### 1. Basic Agent Example (`basic_agent_example.py`)

A simple example showing the basic structure of a JaxABM model with random walker agents.

```bash
python examples/basic_agent_example.py
```

### 2. Minimal Example (`minimal_example.py`)

Minimal demonstration of the core JAX API with a simple random walker model.

```bash
python examples/minimal_example.py
```

### 3. Simple JAX ABM (`jax_abm_simple.py`)

A simplified economic model for quick experimentation, demonstrating:
- Basic producer/consumer agents
- Calibration and sensitivity analysis
- Visualization of results

```bash
# Run the basic simulation
python examples/jax_abm_simple.py

# Run with specific components
python examples/jax_abm_simple.py --simulation --calibration --sensitivity
```

### 4. Standard JAX ABM Example (`jax_abm_example.py`)

A more complete economic model with enhanced features:

```bash
# Run with default settings
python examples/jax_abm_example.py

# Run in fast mode (fewer agents)
python examples/jax_abm_example.py --fast

# Run with calibration
python examples/jax_abm_example.py --calibration

# Run with sensitivity analysis
python examples/jax_abm_example.py --sensitivity
```

### 5. Professional JAX ABM (`jax_abm_professional.py`)

A comprehensive example demonstrating all features of the framework:
- Multiple agent types with complex interactions
- Parameter calibration using gradient-based methods
- Sensitivity analysis with visualization
- Command-line argument handling
- Performance optimizations

```bash
# View available options
python examples/jax_abm_professional.py --help

# Run simulation only
python examples/jax_abm_professional.py --simulation

# Run in fast mode (fewer agents for quick testing)
python examples/jax_abm_professional.py --fast

# Run calibration
python examples/jax_abm_professional.py --calibration

# Run sensitivity analysis
python examples/jax_abm_professional.py --sensitivity

# Skip calibration step in sensitivity analysis
python examples/jax_abm_professional.py --sensitivity --skip-calibration

# Complete run with all components
python examples/jax_abm_professional.py --simulation --calibration --sensitivity --fast
```

## Common Parameters

Most examples support the following common parameters:

- `--fast`: Run with fewer agents and steps for quicker execution
- `--calibration`: Run the model calibration component
- `--sensitivity`: Run the sensitivity analysis component
- `--simulation`: Run the main simulation component
- `--skip-calibration`: Skip the calibration step in sensitivity analysis
- `--seed VALUE`: Set a specific random seed for reproducibility

## Example Output

Examples typically generate console output and may also produce visualization plots showing:
- Economic metrics like GDP, unemployment, prices
- Calibration convergence graphs
- Sensitivity analysis heat maps and Sobol indices
- Agent state distributions 