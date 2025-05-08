"""
Simplified Economic Model using JaxABM

This example implements a simplified version of the economic model 
shown in the diagram. It focuses on households as key agents.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

# Add the parent directory to the path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jax import random

# Import JaxABM components
from jaxabm.agent import AgentType, AgentCollection
from jaxabm.core import ModelConfig
from jaxabm.model import Model

# ===== AGENT TYPES =====

class Household(AgentType):
    """Households that earn income, consume goods, and save money."""
    
    def __init__(self, 
                 initial_savings=1000.0, 
                 initial_income=100.0,
                 propensity_to_consume=0.8):
        """Initialize household parameters."""
        self.initial_savings = initial_savings
        self.initial_income = initial_income
        self.propensity_to_consume = propensity_to_consume
    
    def init_state(self, model_config, key):
        """Initialize household state with some heterogeneity."""
        # Split random key
        key1, key2 = random.split(key)
        
        # Random factors to create heterogeneity
        savings_factor = random.uniform(key1) * 0.5 + 0.75  # 0.75-1.25
        income_factor = random.uniform(key2) * 0.6 + 0.7    # 0.7-1.3
        
        # Initial state
        return {
            'savings': self.initial_savings * savings_factor,
            'income': self.initial_income * income_factor,
            'consumption': 0.0,
            'employed': True,  # Start employed
            'utility': 0.0
        }
    
    def update(self, state, model_state, model_config, key):
        """Update household behavior."""
        # Get environment variables
        wage_rate = model_state['env'].get('wage_rate', 1.0)
        price_level = model_state['env'].get('price_level', 1.0)
        unemployment_rate = model_state['env'].get('unemployment_rate', 0.05)
        
        # Split random key
        key1, key2 = random.split(key)
        
        # Labor market participation - simplified employment dynamics
        # Stay employed with 98% chance, find job with 20% chance if unemployed
        employment_random = random.uniform(key1)
        
        # JAX-compatible conditional
        new_employed = jnp.where(
            state['employed'],
            employment_random > 0.02,  # 98% chance to stay employed
            employment_random < 0.2    # 20% chance to find job
        )
        
        # Calculate income based on employment
        base_income = wage_rate * new_employed
        income_noise = random.uniform(key2) * 0.2 + 0.9  # Random factor 0.9-1.1
        new_income = base_income * income_noise
        
        # Consumption decision (simplified)
        consumption_amount = new_income * self.propensity_to_consume
        
        # Limit consumption by available savings
        actual_consumption = jnp.minimum(consumption_amount, state['savings'])
        
        # Update savings
        new_savings = state['savings'] + new_income - actual_consumption
        
        # Calculate utility (log utility function)
        utility = jnp.log(1.0 + actual_consumption / price_level)
        
        # Return updated state
        return {
            'savings': new_savings,
            'income': new_income,
            'consumption': actual_consumption,
            'employed': new_employed,
            'utility': utility
        }

# ===== ENVIRONMENT FUNCTIONS =====

def update_environment(env_state, agent_states, params, key):
    """Update environmental variables based on agent states."""
    # Get household states
    household_states = agent_states.get('households', {})
    
    if not household_states:  # No households
        return env_state
    
    # Calculate key aggregate metrics
    total_consumption = jnp.sum(household_states.get('consumption', jnp.array([0.0])))
    total_income = jnp.sum(household_states.get('income', jnp.array([0.0])))
    employment_rate = jnp.mean(household_states.get('employed', jnp.array([0.0])).astype(float))
    unemployment_rate = 1.0 - employment_rate
    avg_utility = jnp.mean(household_states.get('utility', jnp.array([0.0])))
    
    # Get current time step
    time_step = env_state.get('time_step', 0) + 1
    
    # Get previous values for calculating changes
    old_price_level = env_state.get('price_level', 1.0)
    old_wage_rate = env_state.get('wage_rate', 1.0)
    old_gdp = env_state.get('gdp', total_consumption)
    
    # Split random key
    key1, key2 = random.split(key)
    
    # Update price level based on consumption demand (simplified)
    # Higher consumption leads to higher prices
    consumption_pressure = total_consumption / (old_gdp + 1e-6)
    price_adjustment = (consumption_pressure - 1.0) * 0.1
    price_noise = random.uniform(key1) * 0.02 - 0.01  # Random factor -0.01 to 0.01
    new_price_level = old_price_level * (1.0 + price_adjustment + price_noise)
    
    # Update wage rate based on unemployment (Phillips curve)
    # Lower unemployment leads to higher wages
    wage_adjustment = (0.05 - unemployment_rate) * 0.2  # 5% is the "natural" rate
    wage_noise = random.uniform(key2) * 0.02 - 0.01  # Random factor -0.01 to 0.01
    new_wage_rate = old_wage_rate * (1.0 + wage_adjustment + wage_noise)
    
    # Calculate GDP (simplified as consumption)
    gdp = total_consumption
    
    # GDP growth rate
    gdp_growth = (gdp / old_gdp) - 1.0 if old_gdp > 0 else 0.0
    
    # Calculate inflation
    inflation_rate = (new_price_level / old_price_level) - 1.0
    
    # Return updated environment state
    return {
        'time_step': time_step,
        'price_level': new_price_level,
        'wage_rate': new_wage_rate,
        'gdp': gdp,
        'gdp_growth': gdp_growth,
        'inflation_rate': inflation_rate,
        'employment_rate': employment_rate,
        'unemployment_rate': unemployment_rate,
        'total_consumption': total_consumption,
        'total_income': total_income,
        'avg_utility': avg_utility
    }

def compute_metrics(env_state, agent_states, params):
    """Compute economic metrics from model state."""
    # Extract key metrics from environment state
    gdp = env_state.get('gdp', 0.0)
    gdp_growth = env_state.get('gdp_growth', 0.0)
    inflation_rate = env_state.get('inflation_rate', 0.0)
    unemployment_rate = env_state.get('unemployment_rate', 0.0)
    avg_utility = env_state.get('avg_utility', 0.0)
    
    # Calculate overall economic health index (simplified)
    economic_health = (
        0.4 * (1.0 - unemployment_rate) +             # Lower unemployment is better
        0.3 * jnp.clip(gdp_growth * 10, -1.0, 1.0) +  # GDP growth contributes positively
        0.2 * (1.0 - jnp.abs(inflation_rate - 0.02) * 10) +  # Target 2% inflation
        0.1 * avg_utility / 2.0                       # Higher utility is better
    )
    
    # Scale to 0-100 range
    economic_health_index = jnp.clip(economic_health * 100, 0, 100)
    
    # Return all metrics
    return {
        'gdp': gdp,
        'gdp_growth': gdp_growth * 100,  # as percentage
        'inflation': inflation_rate * 100,  # as percentage
        'unemployment': unemployment_rate * 100,  # as percentage
        'utility': avg_utility,
        'economic_health': economic_health_index
    }

# ===== MODEL CREATION =====

def create_economy_model(
    num_households=1000,
    propensity_to_consume=0.8,
    initial_wage_rate=1.0,
    initial_price_level=1.0,
    seed=42,
    params=None,
    config=None
):
    """Create the simplified economy model."""
    # Handle params from calibration if provided
    if params is not None:
        propensity_to_consume = params.get('propensity_to_consume', propensity_to_consume)
        initial_wage_rate = params.get('initial_wage_rate', initial_wage_rate)
        initial_price_level = params.get('initial_price_level', initial_price_level)
    
    # Use provided config or create default
    if config is None:
        config = ModelConfig(
            seed=seed,
            steps=50,  # Default simulation length
            track_history=True,
            collect_interval=1  # Collect data every step
        )
    
    # Create model with environment update and metrics functions
    model = Model(
        params=params or {},
        config=config,
        update_state_fn=update_environment,
        metrics_fn=compute_metrics
    )
    
    # Create household agent type and collection
    household_agent_type = Household(
        propensity_to_consume=propensity_to_consume
    )
    
    households = AgentCollection(
        agent_type=household_agent_type,
        num_agents=num_households
    )
    
    # Add households to model
    model.add_agent_collection('households', households)
    
    # Initialize environmental state
    model.add_env_state('time_step', 0)
    model.add_env_state('wage_rate', initial_wage_rate)
    model.add_env_state('price_level', initial_price_level)
    model.add_env_state('gdp', 0.0)
    model.add_env_state('gdp_growth', 0.0)
    model.add_env_state('inflation_rate', 0.02)  # Start with 2% inflation
    model.add_env_state('employment_rate', 0.95)
    model.add_env_state('unemployment_rate', 0.05)
    
    return model

# ===== SIMULATION FUNCTIONS =====

def run_simulation(args):
    """Run the economic simulation."""
    print(f"Running economic simulation with {args.households} households...")
    
    # Create model with command line parameters
    model = create_economy_model(
        num_households=args.households,
        propensity_to_consume=args.propensity,
        seed=args.seed
    )
    
    # Run simulation
    results = model.run(args.steps)
    
    # Display key results
    if results:
        print("\nSimulation completed. Final economic indicators:")
        
        # Get final values
        final_idx = -1  # Last time step
        
        print(f"GDP: {results['gdp'][final_idx]:.4f}")
        print(f"GDP Growth: {results['gdp_growth'][final_idx]:.2f}%")
        print(f"Unemployment: {results['unemployment'][final_idx]:.2f}%")
        print(f"Inflation: {results['inflation'][final_idx]:.2f}%")
        print(f"Avg Utility: {results['utility'][final_idx]:.4f}")
        print(f"Economic Health Index: {results['economic_health'][final_idx]:.1f}/100")
        
        # Plot results
        plot_results(results)
    
    return results

# ===== VISUALIZATION =====

def plot_results(results, title="Economic Simulation Results"):
    """Plot key economic indicators from simulation results."""
    # Check if we have results to plot
    if not results or 'step' not in results:
        print("No results to plot")
        return
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(title, fontsize=16)
    
    # Time steps
    time_steps = results['step']
    
    # Plot 1: GDP and Growth
    ax1 = axs[0, 0]
    ax1.plot(time_steps, results['gdp'], 'b-', label='GDP')
    ax1.set_ylabel('GDP', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax1b = ax1.twinx()
    ax1b.plot(time_steps, results['gdp_growth'], 'r-', label='Growth %')
    ax1b.set_ylabel('Growth %', color='r')
    ax1b.tick_params(axis='y', labelcolor='r')
    ax1.set_title('Economic Output')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Unemployment and Inflation
    ax2 = axs[0, 1]
    ax2.plot(time_steps, results['unemployment'], 'g-', label='Unemployment %')
    ax2.set_ylabel('Unemployment %', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    
    ax2b = ax2.twinx()
    ax2b.plot(time_steps, results['inflation'], 'm-', label='Inflation %')
    ax2b.set_ylabel('Inflation %', color='m')
    ax2b.tick_params(axis='y', labelcolor='m')
    ax2.set_title('Labor Market & Prices')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Utility
    ax3 = axs[1, 0]
    ax3.plot(time_steps, results['utility'], 'c-')
    ax3.set_ylabel('Average Utility')
    ax3.set_title('Household Welfare')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Economic Health Index
    ax4 = axs[1, 1]
    ax4.plot(time_steps, results['economic_health'], 'k-')
    ax4.set_ylabel('Index (0-100)')
    ax4.set_title('Economic Health Index')
    ax4.grid(True, alpha=0.3)
    
    # Adjust layout and save/show
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    plt.savefig('economic_simulation_results.png', dpi=100, bbox_inches='tight')
    
    print("Plot saved as 'economic_simulation_results.png'")
    
    # Display plot
    plt.show()

# ===== MAIN FUNCTION =====

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simplified economic model simulation")
    parser.add_argument("--households", type=int, default=1000, help="Number of households")
    parser.add_argument("--steps", type=int, default=50, help="Number of simulation steps")
    parser.add_argument("--propensity", type=float, default=0.8, help="Propensity to consume")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_simulation(args) 