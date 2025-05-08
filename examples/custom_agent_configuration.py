#!/usr/bin/env python3
"""
Custom Agent Configuration Example

This example demonstrates how to configure the economic model with different
numbers of agent types (households, consumer firms, capital firms, energy firms)
and compares the results of different economic configurations.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our model
from examples.advanced_economic_model import create_economy_model

# Define different economic configurations to test
CONFIGURATIONS = [
    {
        "name": "Balanced Economy",
        "households": 1000,
        "consumer_firms": 50,
        "capital_firms": 20,
        "energy_firms": 10,
        "color": "blue"
    },
    {
        "name": "Service Economy",
        "households": 1000,
        "consumer_firms": 80,
        "capital_firms": 10,
        "energy_firms": 5,
        "color": "green"
    },
    {
        "name": "Industrial Economy",
        "households": 1000,
        "consumer_firms": 30,
        "capital_firms": 40,
        "energy_firms": 15,
        "color": "red"
    },
    {
        "name": "Green Economy",
        "households": 1000,
        "consumer_firms": 40,
        "capital_firms": 20,
        "energy_firms": 25,
        "energy_firm_params": {"renewable_fraction": 0.5},
        "color": "green"
    }
]

def run_simulation(config: Dict[str, Any], seed: int = 42, steps: int = 100):
    """Run a simulation with the given configuration."""
    
    print(f"Running simulation: {config['name']}")
    
    # Create model with specified agent counts
    model = create_economy_model(
        num_households=config.get("households", 1000),
        num_consumer_firms=config.get("consumer_firms", 50),
        num_capital_firms=config.get("capital_firms", 20),
        num_energy_firms=config.get("energy_firms", 10),
        # Pass through any specialized parameters
        household_params=config.get("household_params", None),
        consumer_firm_params=config.get("consumer_firm_params", None),
        capital_firm_params=config.get("capital_firm_params", None),
        energy_firm_params=config.get("energy_firm_params", None),
        enable_climate_module=config.get("enable_climate", False),
        enable_pandemic_module=config.get("enable_pandemic", False),
        seed=seed
    )
    
    # Set model config steps
    model.config.steps = steps
    
    # Run the simulation
    history = model.run()
    
    # Return the history for analysis
    return history

def compare_configurations():
    """Run simulations with different configurations and compare results."""
    
    # Dictionary to store histories for each configuration
    histories = {}
    
    # Run simulations for each configuration
    for config in CONFIGURATIONS:
        histories[config["name"]] = run_simulation(config)
    
    # Plot key metrics to compare configurations
    plot_comparative_metrics(histories)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 80)
    print(f"{'Configuration':<20} {'Final GDP':<12} {'GDP Growth':<12} {'Unemployment':<12} {'Sustainability':<12}")
    print("-" * 80)
    
    for name, history in histories.items():
        # Get final values
        final_gdp = history["gdp"][-1]
        # Calculate average GDP growth over the last 10 periods
        gdp_growth = np.mean(history["gdp_growth"][-10:])
        unemployment = history["unemployment"][-1]
        sustainability = history["sustainability_index"][-1] if "sustainability_index" in history else 0
        
        print(f"{name:<20} {final_gdp:<12.2f} {gdp_growth:<12.2f} {unemployment:<12.2f} {sustainability:<12.2f}")

def plot_comparative_metrics(histories):
    """Create comparative plots of key metrics across configurations."""
    
    # Define metrics to plot
    metrics_to_plot = [
        {"name": "gdp", "title": "GDP Over Time", "ylabel": "GDP"},
        {"name": "unemployment", "title": "Unemployment Rate", "ylabel": "Unemployment %"},
        {"name": "economic_health", "title": "Economic Health Index", "ylabel": "Index Value"},
        {"name": "sustainability_index", "title": "Sustainability Index", "ylabel": "Index Value"}
    ]
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot each metric
    for i, metric_info in enumerate(metrics_to_plot):
        metric_name = metric_info["name"]
        
        for config in CONFIGURATIONS:
            name = config["name"]
            color = config.get("color", "blue")
            history = histories[name]
            
            if metric_name in history:
                axes[i].plot(
                    history[metric_name], 
                    label=name,
                    color=color,
                    alpha=0.8
                )
        
        axes[i].set_title(metric_info["title"])
        axes[i].set_xlabel("Time Steps")
        axes[i].set_ylabel(metric_info["ylabel"])
        axes[i].grid(True, alpha=0.3)
        
        # Only add legend to the first plot to avoid clutter
        if i == 0:
            axes[i].legend()
    
    plt.tight_layout()
    plt.savefig("economy_configuration_comparison.png")
    print("\nComparison plot saved as 'economy_configuration_comparison.png'")
    plt.close()
    
    # Create additional sector-specific plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Define sector metrics to plot
    sector_metrics = [
        {"name": "consumer_sector_share", "title": "Consumer Sector Share of GDP", "ylabel": "%"},
        {"name": "capital_sector_share", "title": "Capital Sector Share of GDP", "ylabel": "%"},
        {"name": "energy_sector_share", "title": "Energy Sector Share of GDP", "ylabel": "%"},
        {"name": "renewable_share", "title": "Renewable Energy Share", "ylabel": "%"}
    ]
    
    # Plot sector metrics
    for i, metric_info in enumerate(sector_metrics):
        metric_name = metric_info["name"]
        
        for config in CONFIGURATIONS:
            name = config["name"]
            color = config.get("color", "blue")
            history = histories[name]
            
            if metric_name in history:
                axes[i].plot(
                    history[metric_name], 
                    label=name,
                    color=color,
                    alpha=0.8
                )
        
        axes[i].set_title(metric_info["title"])
        axes[i].set_xlabel("Time Steps")
        axes[i].set_ylabel(metric_info["ylabel"])
        axes[i].grid(True, alpha=0.3)
        
        # Add legend to all sector plots
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig("economy_sector_comparison.png")
    print("Sector comparison plot saved as 'economy_sector_comparison.png'")

if __name__ == "__main__":
    print("Comparing different economic configurations with varying agent counts\n")
    compare_configurations() 