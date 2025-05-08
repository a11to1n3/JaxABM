# Advanced Economic Model

This module provides a sophisticated agent-based macroeconomic model that simulates a complex economy with multiple interacting agent types.

## Features

- Multiple agent types (Households, different Firm types, Government, Banks)
- Complex economic flows and feedback loops
- Environmental and external shock modeling
- Policy intervention testing
- Easy configuration system for customizing simulations

## Agent Types

The model includes the following agent types:

- **Households**: Provide labor, consume goods, pay taxes, save money
- **Consumer Goods Firms**: Produce goods for household consumption
- **Capital Goods Firms**: Produce capital for other firms
- **Energy Firms**: Produce energy needed for production
- **Government**: Collects taxes, issues bonds, provides transfers
- **Banks**: Manage savings and provide credit

## Configuration System

The new configuration system makes it easy to create custom simulations with different agent populations and parameters.

### Basic Usage

```python
from examples.economy_model_configuration import EconomyConfiguration

# Create a custom economy
config = EconomyConfiguration(
    num_households=1000,
    num_consumer_firms=50,
    num_capital_firms=20,
    num_energy_firms=10,
    enable_climate_module=True
)

# Run the simulation
results = config.run_simulation()
```

### Using Presets

Several preset configurations are available for common economic structures:

```python
from examples.economy_model_configuration import EconomyPresets

# Get a small economy configuration (good for testing)
small_config = EconomyPresets.small_economy()

# Get a service-oriented economy
service_config = EconomyPresets.service_economy()

# Get an industrial economy
industrial_config = EconomyPresets.industrial_economy()

# Get a crisis economy (with climate and pandemic modules)
crisis_config = EconomyPresets.crisis_economy()

# Run any of these
results = service_config.run_simulation()
```

### Customizing Agent Parameters

You can customize the parameters for any agent type:

```python
# Custom household parameters
config = EconomyConfiguration(
    num_households=500,
    household_params={
        'initial_savings': 2000.0,
        'initial_income': 150.0,
        'propensity_to_consume': 0.6,
        'propensity_to_save': 0.3,
        'risk_aversion': 0.7
    }
)

# Custom firm parameters
config = EconomyConfiguration(
    num_consumer_firms=30,
    consumer_firm_params={
        'initial_capital': 1500.0,
        'production_efficiency': 1.2,
        'markup_rate': 0.15
    }
)
```

### Saving and Loading Configurations

Configurations can be saved to and loaded from JSON files:

```python
# Save configuration
config.save_to_file("my_economy.json")

# Load configuration
from examples.economy_model_configuration import EconomyConfiguration
loaded_config = EconomyConfiguration.load_from_file("my_economy.json")
```

## Command Line Interface

The configuration system includes a command-line interface for running simulations:

```bash
# Run with default settings
python examples/economy_model_configuration.py --run

# Use a preset
python examples/economy_model_configuration.py --preset industrial --run

# Customize agent populations
python examples/economy_model_configuration.py --households 2000 --consumer-firms 100 --run

# Enable external modules
python examples/economy_model_configuration.py --climate --pandemic --run

# Save and load configurations
python examples/economy_model_configuration.py --households 1500 --save-config my_config.json
python examples/economy_model_configuration.py --load-config my_config.json --run
```

## Examples

See `examples/custom_economy_example.py` for more detailed examples of how to use the configuration system, including:

1. Comparing different economic structures
2. Using preset configurations
3. Customizing household parameters
4. Customizing firm parameters
5. Saving and loading configurations

## Advanced Features

- **Sensitivity Analysis**: Test how the economy responds to changes in key parameters
- **Policy Experiments**: Test different policy interventions
- **Shock Resilience Testing**: Evaluate economic resilience to various shocks

## Visualization

The system includes built-in visualization tools for comparing economic outcomes:

```python
# Run multiple configurations and compare
from examples.custom_economy_example import compare_different_economies
results = compare_different_economies()
```

This will generate comparative plots showing how different economic structures perform on metrics like GDP growth, unemployment, inflation, and overall economic health. 