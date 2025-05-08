"""
Economy Model Configuration

This module provides a configuration system for the advanced economic model.
It allows easy customization of:
- Agent populations (households, different firm types)
- Agent parameters 
- Environmental settings
- External modules (climate, pandemic)
- Simulation settings

Use this for creating custom economic simulations with different agent compositions.
"""

import os
import sys
import argparse
import json
from typing import Dict, Any, Optional, List, Tuple

# Add the parent directory to the path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaxabm.core import ModelConfig

# Import model creation function 
from examples.advanced_economic_model import create_economy_model, run_simulation

class EconomyConfiguration:
    """Configuration class for the advanced economic model.
    
    This class provides a structured way to configure all aspects
    of the economic model, including:
    - Agent populations and parameters
    - Economic environment settings
    - External impact modules
    - Simulation parameters
    """
    
    def __init__(self, 
                 # Agent populations
                 num_households: int = 1000,
                 num_consumer_firms: int = 50,
                 num_capital_firms: int = 20, 
                 num_energy_firms: int = 10,
                 num_banks: int = 5,
                 num_govt_agents: int = 1,
                 
                 # Economic parameters
                 tax_rate: float = 0.2,
                 interest_rate: float = 0.05,
                 energy_price: float = 1.0,
                 wage_rate: float = 1.0,
                 
                 # External modules
                 enable_climate_module: bool = False,
                 enable_pandemic_module: bool = False,
                 
                 # Agent params (optional dicts for customizing agent parameters)
                 household_params: Optional[Dict[str, Any]] = None,
                 consumer_firm_params: Optional[Dict[str, Any]] = None,
                 capital_firm_params: Optional[Dict[str, Any]] = None,
                 energy_firm_params: Optional[Dict[str, Any]] = None,
                 bank_params: Optional[Dict[str, Any]] = None,
                 govt_params: Optional[Dict[str, Any]] = None,
                 
                 # Simulation settings
                 num_steps: int = 100,
                 seed: int = 42,
                 collect_interval: int = 1):
        """Initialize economy configuration."""
        # Agent populations
        self.num_households = num_households
        self.num_consumer_firms = num_consumer_firms
        self.num_capital_firms = num_capital_firms
        self.num_energy_firms = num_energy_firms
        self.num_banks = num_banks
        self.num_govt_agents = num_govt_agents
        
        # Economic parameters
        self.tax_rate = tax_rate
        self.interest_rate = interest_rate
        self.energy_price = energy_price
        self.wage_rate = wage_rate
        
        # External modules
        self.enable_climate_module = enable_climate_module
        self.enable_pandemic_module = enable_pandemic_module
        
        # Setup default agent parameters if not provided
        self.household_params = household_params or {
            'initial_savings': 1000.0,
            'initial_income': 100.0,
            'propensity_to_consume': 0.8,
            'propensity_to_save': 0.1,
            'labor_productivity': 1.0,
            'risk_aversion': 0.5
        }
        
        self.consumer_firm_params = consumer_firm_params or {
            'initial_capital': 1000.0, 
            'initial_cash': 500.0,
            'production_efficiency': 1.0,
            'labor_elasticity': 0.6,
            'capital_elasticity': 0.3,
            'energy_elasticity': 0.1,
            'markup_rate': 0.2
        }
        
        self.capital_firm_params = capital_firm_params or {}
        self.energy_firm_params = energy_firm_params or {}
        self.bank_params = bank_params or {}
        self.govt_params = govt_params or {}
        
        # Simulation parameters
        self.num_steps = num_steps
        self.seed = seed
        self.collect_interval = collect_interval
    
    def as_dict(self) -> Dict[str, Any]:
        """Return configuration as a dictionary."""
        return {
            # Agent populations
            'num_households': self.num_households,
            'num_consumer_firms': self.num_consumer_firms,
            'num_capital_firms': self.num_capital_firms,
            'num_energy_firms': self.num_energy_firms,
            'num_banks': self.num_banks,
            'num_govt_agents': self.num_govt_agents,
            
            # Economic parameters
            'tax_rate': self.tax_rate,
            'interest_rate': self.interest_rate,
            'energy_price': self.energy_price,
            'wage_rate': self.wage_rate,
            
            # External modules
            'enable_climate_module': self.enable_climate_module,
            'enable_pandemic_module': self.enable_pandemic_module,
            
            # Agent parameters
            'household_params': self.household_params,
            'consumer_firm_params': self.consumer_firm_params,
            'capital_firm_params': self.capital_firm_params,
            'energy_firm_params': self.energy_firm_params,
            'bank_params': self.bank_params,
            'govt_params': self.govt_params,
            
            # Simulation parameters
            'num_steps': self.num_steps,
            'seed': self.seed,
            'collect_interval': self.collect_interval
        }
    
    def save_to_file(self, filename: str) -> None:
        """Save configuration to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.as_dict(), f, indent=2)
        print(f"Configuration saved to {filename}")
    
    @classmethod
    def load_from_file(cls, filename: str) -> 'EconomyConfiguration':
        """Load configuration from a JSON file."""
        with open(filename, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def create_model(self):
        """Create an economic model based on this configuration."""
        # Create model config
        model_config = ModelConfig(
            seed=self.seed,
            steps=self.num_steps,
            track_history=True,
            collect_interval=self.collect_interval
        )
        
        # Create and return model
        return create_economy_model(
            num_households=self.num_households,
            num_consumer_firms=self.num_consumer_firms,
            num_capital_firms=self.num_capital_firms,
            num_energy_firms=self.num_energy_firms,
            enable_climate_module=self.enable_climate_module,
            enable_pandemic_module=self.enable_pandemic_module,
            tax_rate=self.tax_rate,
            interest_rate=self.interest_rate,
            energy_price=self.energy_price,
            wage_rate=self.wage_rate,
            household_params=self.household_params,
            consumer_firm_params=self.consumer_firm_params,
            capital_firm_params=self.capital_firm_params,
            energy_firm_params=self.energy_firm_params,
            govt_params=self.govt_params,
            bank_params=self.bank_params,
            seed=self.seed,
            config=model_config
        )
    
    def run_simulation(self):
        """Create and run a simulation with this configuration."""
        model = self.create_model()
        return model.run()

# Preset configurations for common economic structures
class EconomyPresets:
    """Collection of preset economy configurations for different scenarios."""
    
    @staticmethod
    def small_economy() -> EconomyConfiguration:
        """Small economy with few agents (good for testing)."""
        return EconomyConfiguration(
            num_households=100,
            num_consumer_firms=10,
            num_capital_firms=5,
            num_energy_firms=2,
            num_banks=1,
            num_govt_agents=1,
            num_steps=50,
            seed=42
        )
    
    @staticmethod
    def standard_economy() -> EconomyConfiguration:
        """Standard balanced economy with default parameters."""
        return EconomyConfiguration(
            num_households=1000,
            num_consumer_firms=50,
            num_capital_firms=20,
            num_energy_firms=10,
            num_banks=5,
            num_govt_agents=1,
            num_steps=100,
            seed=42
        )
    
    @staticmethod
    def service_economy() -> EconomyConfiguration:
        """Service-oriented economy with more households and consumer firms."""
        return EconomyConfiguration(
            num_households=2000,
            num_consumer_firms=100,
            num_capital_firms=10,
            num_energy_firms=5,
            num_banks=10,
            num_govt_agents=1,
            household_params={
                'propensity_to_consume': 0.85,
                'propensity_to_save': 0.05
            },
            num_steps=100,
            seed=42
        )
    
    @staticmethod
    def industrial_economy() -> EconomyConfiguration:
        """Industrial economy with emphasis on capital goods production."""
        return EconomyConfiguration(
            num_households=1500,
            num_consumer_firms=40,
            num_capital_firms=60,
            num_energy_firms=20,
            num_banks=5,
            num_govt_agents=1,
            household_params={
                'propensity_to_consume': 0.7,
                'propensity_to_save': 0.2
            },
            num_steps=100,
            seed=42
        )
    
    @staticmethod
    def crisis_economy() -> EconomyConfiguration:
        """Economy with pandemic and climate shocks active."""
        return EconomyConfiguration(
            num_households=1000,
            num_consumer_firms=50,
            num_capital_firms=20,
            num_energy_firms=10,
            enable_climate_module=True,
            enable_pandemic_module=True,
            num_steps=150,  # Longer run to see impact and recovery
            seed=42
        )

def parse_args():
    """Parse command line arguments for configuration."""
    parser = argparse.ArgumentParser(description="Configure and run economic model simulation")
    
    # Agent population settings
    population_group = parser.add_argument_group("Agent populations")
    population_group.add_argument("--households", type=int, default=1000, help="Number of households")
    population_group.add_argument("--consumer-firms", type=int, default=50, help="Number of consumer goods firms")
    population_group.add_argument("--capital-firms", type=int, default=20, help="Number of capital goods firms")
    population_group.add_argument("--energy-firms", type=int, default=10, help="Number of energy firms")
    population_group.add_argument("--banks", type=int, default=5, help="Number of banks")
    
    # Economic parameter settings
    econ_group = parser.add_argument_group("Economic parameters")
    econ_group.add_argument("--tax-rate", type=float, default=0.2, help="Tax rate")
    econ_group.add_argument("--interest-rate", type=float, default=0.05, help="Interest rate")
    econ_group.add_argument("--energy-price", type=float, default=1.0, help="Initial energy price")
    
    # External modules
    modules_group = parser.add_argument_group("External modules")
    modules_group.add_argument("--climate", action="store_true", help="Enable climate module")
    modules_group.add_argument("--pandemic", action="store_true", help="Enable pandemic module")
    
    # Simulation settings
    sim_group = parser.add_argument_group("Simulation settings")
    sim_group.add_argument("--steps", type=int, default=100, help="Number of simulation steps")
    sim_group.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Presets and file options
    preset_group = parser.add_argument_group("Presets and file options")
    preset_group.add_argument("--preset", choices=["small", "standard", "service", "industrial", "crisis"],
                            help="Use a preset configuration")
    preset_group.add_argument("--save-config", type=str, help="Save configuration to file")
    preset_group.add_argument("--load-config", type=str, help="Load configuration from file")
    
    # Run options
    run_group = parser.add_argument_group("Run options")
    run_group.add_argument("--run", action="store_true", help="Run simulation after configuration")
    run_group.add_argument("--sensitivity", action="store_true", help="Run sensitivity analysis")
    run_group.add_argument("--policy", action="store_true", help="Run policy experiment")
    run_group.add_argument("--shock", action="store_true", help="Run shock resilience experiment")
    
    return parser.parse_args()

def main():
    """Main function to run from command line."""
    args = parse_args()
    
    # Load configuration from preset or file, or use command line arguments
    if args.preset:
        if args.preset == "small":
            config = EconomyPresets.small_economy()
        elif args.preset == "standard":
            config = EconomyPresets.standard_economy()
        elif args.preset == "service":
            config = EconomyPresets.service_economy()
        elif args.preset == "industrial":
            config = EconomyPresets.industrial_economy()
        elif args.preset == "crisis":
            config = EconomyPresets.crisis_economy()
    elif args.load_config:
        config = EconomyConfiguration.load_from_file(args.load_config)
    else:
        # Create configuration from command line arguments
        config = EconomyConfiguration(
            num_households=args.households,
            num_consumer_firms=args.consumer_firms,
            num_capital_firms=args.capital_firms,
            num_energy_firms=args.energy_firms,
            num_banks=args.banks,
            tax_rate=args.tax_rate,
            interest_rate=args.interest_rate,
            energy_price=args.energy_price,
            enable_climate_module=args.climate,
            enable_pandemic_module=args.pandemic,
            num_steps=args.steps,
            seed=args.seed
        )
    
    # Save configuration if requested
    if args.save_config:
        config.save_to_file(args.save_config)
    
    # Print configuration summary
    print("Economy Configuration:")
    print(f"  Households: {config.num_households}")
    print(f"  Consumer Firms: {config.num_consumer_firms}")
    print(f"  Capital Goods Firms: {config.num_capital_firms}")
    print(f"  Energy Firms: {config.num_energy_firms}")
    print(f"  Banks: {config.num_banks}")
    print(f"  Climate Module: {'Enabled' if config.enable_climate_module else 'Disabled'}")
    print(f"  Pandemic Module: {'Enabled' if config.enable_pandemic_module else 'Disabled'}")
    print(f"  Simulation Steps: {config.num_steps}")
    
    # Run simulation if requested
    if args.run:
        print("\nRunning simulation...")
        results = config.run_simulation()
        
        # Display final results
        if results:
            final_idx = -1  # Last time step
            print("\nSimulation completed. Final economic indicators:")
            print(f"GDP: {results['gdp'][final_idx]:.4f}")
            print(f"GDP Growth: {results['gdp_growth'][final_idx]:.2f}%")
            print(f"Unemployment: {results['unemployment'][final_idx]:.2f}%")
            print(f"Inflation: {results['inflation'][final_idx]:.2f}%")
            print(f"Income Inequality (Gini): {results['inequality'][final_idx]:.4f}")
            print(f"Economic Health Index: {results['economic_health'][final_idx]:.1f}/100")
    
    # TODO: Add support for sensitivity, policy, and shock experiments based on args
    
    return config

if __name__ == "__main__":
    main() 