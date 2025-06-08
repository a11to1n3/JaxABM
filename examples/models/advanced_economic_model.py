"""
Advanced Economic Model using JaxABM

This example implements a complex macroeconomic model that demonstrates the full
capabilities of the JaxABM framework. The model includes:
- Households (providing labor and consumption)
- Multiple firm types (Consumer Goods, Capital Goods, Energy)
- Government (fiscal policy, transfers, bonds)
- Banking sector (savings, credit, bonds)
- External shock modules (Climate, Pandemic)

Model flow diagram:
- Households provide labor to firms and receive wages
- Households consume goods, save money in banks
- Firms produce goods using labor, capital, and energy
- Government collects taxes, issues bonds, provides transfers
- Banks manage savings and provide credit
- External modules influence the economy through shocks

This demonstrates advanced features of JaxABM including:
- Multiple interacting agent types
- Complex economic flows and feedback loops
- Environmental and external shock modeling
- Policy interventions and testing

The model can be used for:
- Exploring economic resilience to various shocks
- Testing policy interventions
- Analyzing transition pathways (e.g., to green energy)
- Macroeconomic forecasting
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from typing import Dict, Any, List, Tuple

# Add the parent directory to the path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jax import random

# Import JaxABM components
from jaxabm.agent import AgentType, AgentCollection
from jaxabm.core import ModelConfig
from jaxabm.model import Model
from jaxabm.analysis import SensitivityAnalysis, ModelCalibrator

# ------------- AGENT TYPES -------------

# === Household Agent ===
class Household(AgentType):
    """Households provide labor, consume goods, pay taxes, and save money."""
    
    def __init__(self, 
                 initial_savings=1000.0, 
                 initial_income=100.0,
                 propensity_to_consume=0.8,
                 propensity_to_save=0.1,
                 labor_productivity=1.0,
                 risk_aversion=0.5):
        """Initialize household parameters.
        
        Args:
            initial_savings: Starting amount of savings
            initial_income: Starting income level
            propensity_to_consume: Fraction of income spent on consumption
            propensity_to_save: Fraction of income saved in banks
            labor_productivity: Base productivity of labor
            risk_aversion: Determines investment vs savings behavior
        """
        self.initial_savings = initial_savings
        self.initial_income = initial_income
        self.propensity_to_consume = propensity_to_consume
        self.propensity_to_save = propensity_to_save
        self.labor_productivity = labor_productivity
        self.risk_aversion = risk_aversion
    
    def init_state(self, model_config, key):
        """Initialize household state.
        
        Creates the initial state for each household agent, including:
        - Savings (cash + bank deposits)
        - Income (from labor)
        - Employment status
        - Consumption needs
        - Tax obligations
        """
        # Add some heterogeneity to households
        subkey1, subkey2, subkey3, subkey4 = random.split(key, 4)
        
        # Randomize initial savings (lognormal distribution)
        savings_factor = random.lognormal(subkey1, sigma=0.5) 
        initial_savings = self.initial_savings * savings_factor
        
        # Randomize income (normal distribution around mean)
        income_factor = jnp.maximum(0.3, random.normal(subkey2) * 0.2 + 1.0)
        initial_income = self.initial_income * income_factor
        
        # Randomize consumption preferences (beta distribution)
        consume_adj = random.beta(subkey3, a=5, b=2) * 0.4 + 0.6  # Range ~0.6-1.0
        propensity_to_consume = self.propensity_to_consume * consume_adj
        
        # Initialize employed status (most start employed)
        employed = random.uniform(subkey4) < 0.95
        
        # Create initial state
        return {
            # Financial state
            'savings': initial_savings,
            'income': initial_income,
            'bank_deposits': initial_savings * 0.7,  # 70% of savings in bank
            'cash': initial_savings * 0.3,  # 30% of savings as cash
            'debt': 0.0,
            
            # Economic behavior
            'propensity_to_consume': propensity_to_consume,
            'propensity_to_save': self.propensity_to_save,
            'risk_aversion': self.risk_aversion,
            
            # Labor market status
            'employed': employed,
            'productivity': self.labor_productivity * income_factor,
            'labor_supply': employed * 1.0,  # 1.0 unit if employed, 0 if not
            
            # Consumption tracking
            'consumption': 0.0,
            'utility': 0.0,
            'taxes_paid': 0.0,
            'transfers_received': 0.0
        }
    
    def update(self, state, model_state, model_config, key):
        """Update household behavior: work, consume, save, pay taxes.
        
        The household agent performs the following actions each step:
        1. Supply labor to firms if employed
        2. Receive income (wages if employed, transfers if not)
        3. Pay taxes to government
        4. Decide consumption allocation
        5. Save/withdraw from bank
        6. Calculate utility
        
        Args:
            state: Current agent state
            model_state: Current model state including all environmental variables
            model_config: Model configuration parameters
            key: Random key for stochastic decisions
            
        Returns:
            Updated agent state
        """
        # Split random key for various stochastic decisions
        subkey1, subkey2, subkey3, subkey4 = random.split(key, 4)
        
        # --- STEP 1: Labor Market Participation ---
        
        # Check for exogenous employment shocks (pandemic, etc.)
        job_market_condition = model_state['env'].get('job_market_condition', 1.0)
        pandemic_factor = model_state['env'].get('pandemic_impact', 1.0)
        employment_shock = job_market_condition * pandemic_factor
        
        # Update employment status (with persistence and labor market conditions)
        # Calculate probabilities based on current status
        job_loss_prob = 0.02 / employment_shock  # Higher during crisis
        job_find_prob = 0.1 * employment_shock  # Lower during crisis
        
        # JAX-friendly conditional using where instead of if/else
        random_value = random.uniform(subkey1)
        new_employed = jnp.where(
            state['employed'],
            # If employed: stay employed if random > job_loss_prob
            random_value > job_loss_prob,
            # If unemployed: become employed if random < job_find_prob
            random_value < job_find_prob
        )
        
        # Labor supply is 1 if employed, 0 if unemployed
        labor_supply = new_employed * 1.0
        
        # --- STEP 2: Income Calculation ---
        
        # Base income calculation
        wage_rate = model_state['env'].get('wage_rate', 1.0)
        tax_rate = model_state['env'].get('tax_rate', 0.2)
        
        # Calculate income from work
        labor_income = labor_supply * state['productivity'] * wage_rate
        
        # Calculate transfers (unemployment benefits)
        transfer_rate = model_state['env'].get('transfer_rate', 0.5)
        transfers = (1.0 - new_employed) * state['income'] * transfer_rate
        
        # Total gross income
        gross_income = labor_income + transfers
        
        # --- STEP 3: Taxation ---
        
        # Calculate taxes (progressive tax model)
        base_tax = gross_income * tax_rate
        # Progressive component - higher incomes taxed more
        progressive_adj = jnp.where(gross_income > 2 * self.initial_income,
                                    0.05 * (gross_income / self.initial_income - 2), 
                                    0.0)
        taxes = base_tax * (1.0 + progressive_adj)
        
        # Net income after taxes
        net_income = gross_income - taxes
        
        # --- STEP 4: Consumption Decisions ---
        
        # Get current price levels
        price_level = model_state['env'].get('price_level', 1.0)
        goods_availability = model_state['env'].get('goods_availability', 1.0)
        
        # Basic consumption based on propensity to consume
        desired_consumption = net_income * state['propensity_to_consume']
        
        # Check if household has enough money (cash + accessible savings)
        available_funds = state['cash'] + state['bank_deposits'] * 0.3  # Assume 30% of deposits are liquid
        
        # Actual consumption limited by available funds and goods availability
        actual_consumption = jnp.minimum(
            desired_consumption,
            available_funds
        ) * goods_availability
        
        # Real consumption (adjusted for price level)
        real_consumption = actual_consumption / price_level
        
        # --- STEP 5: Savings Decisions ---
        
        # Interest rate from banks
        interest_rate = model_state['env'].get('interest_rate', 0.01)
        
        # Calculate interest income from existing deposits
        interest_income = state['bank_deposits'] * interest_rate
        
        # Decision to save or withdraw
        savings_target = net_income * state['propensity_to_save']
        savings_adjustment = savings_target - state['bank_deposits'] * 0.1  # Gradual adjustment
        
        # New deposits (or withdrawals if negative)
        deposit_change = jnp.minimum(savings_adjustment, state['cash'] - actual_consumption)
        
        # --- STEP 6: Update Financial Position ---
        
        # Update cash position
        new_cash = state['cash'] - actual_consumption - jnp.maximum(0, deposit_change) + jnp.maximum(0, -deposit_change)
        
        # Update bank deposits
        new_bank_deposits = state['bank_deposits'] + jnp.maximum(0, deposit_change) + interest_income
        
        # --- STEP 7: Calculate Utility ---
        
        # Simple utility function based on consumption and savings
        # Log utility for consumption with diminishing returns
        consumption_utility = jnp.log1p(real_consumption)
        
        # Safety utility from savings (risk aversion factor)
        savings_utility = state['risk_aversion'] * jnp.log1p(new_bank_deposits / 100)
        
        # Calculate total utility
        utility = consumption_utility + savings_utility
        
        # --- STEP 8: Update State ---
        
        return {
            # Financial state
            'savings': new_cash + new_bank_deposits,
            'income': gross_income,
            'bank_deposits': new_bank_deposits,
            'cash': new_cash,
            'debt': state['debt'],  # No debt change in this simplified model
            
            # Keep economic behavior parameters
            'propensity_to_consume': state['propensity_to_consume'],
            'propensity_to_save': state['propensity_to_save'],
            'risk_aversion': state['risk_aversion'],
            
            # Updated labor market status
            'employed': new_employed,
            'productivity': state['productivity'],
            'labor_supply': labor_supply,
            
            # Updated consumption tracking
            'consumption': real_consumption,
            'utility': utility,
            'taxes_paid': taxes,
            'transfers_received': transfers
        }

# === Firm Agents ===
class ConsumerGoodsFirm(AgentType):
    """Firms that produce consumer goods using labor, capital, and energy."""
    
    def __init__(self, 
                initial_capital=1000.0, 
                initial_cash=500.0,
                production_efficiency=1.0,
                labor_elasticity=0.6,
                capital_elasticity=0.3,
                energy_elasticity=0.1,
                markup_rate=0.2):
        """Initialize consumer goods firm parameters.
        
        Args:
            initial_capital: Starting capital stock
            initial_cash: Starting cash reserves
            production_efficiency: Total factor productivity
            labor_elasticity: Output elasticity of labor
            capital_elasticity: Output elasticity of capital
            energy_elasticity: Output elasticity of energy
            markup_rate: Profit margin added to production costs
        """
        self.initial_capital = initial_capital
        self.initial_cash = initial_cash
        self.production_efficiency = production_efficiency
        self.labor_elasticity = labor_elasticity
        self.capital_elasticity = capital_elasticity
        self.energy_elasticity = energy_elasticity
        self.markup_rate = markup_rate
    
    def init_state(self, model_config, key):
        """Initialize consumer goods firm state.
        
        Creates the initial state for each firm, including:
        - Capital stock
        - Cash reserves
        - Production capacity
        - Labor demand
        - Energy usage
        """
        # Add heterogeneity to firms
        subkey1, subkey2, subkey3 = random.split(key, 3)
        
        # Randomize initial capital (lognormal distribution)
        capital_factor = random.lognormal(subkey1, sigma=0.5)
        initial_capital = self.initial_capital * capital_factor
        
        # Randomize efficiency (normal distribution around mean)
        efficiency_factor = jnp.maximum(0.5, random.normal(subkey2) * 0.2 + 1.0)
        production_efficiency = self.production_efficiency * efficiency_factor
        
        # Randomize markup (beta distribution)
        markup_adj = random.beta(subkey3, a=5, b=2) * 0.3 + 0.1  # Range ~0.1-0.4
        markup_rate = self.markup_rate * markup_adj
        
        # Calculate initial production capacity based on capital
        # (Assumes default levels of other inputs)
        production_capacity = production_efficiency * (initial_capital ** self.capital_elasticity)
        
        return {
            # Physical capital
            'capital_stock': initial_capital,
            'production_capacity': production_capacity,
            'inventory': 0.0,
            
            # Financial state
            'cash': self.initial_cash,
            'revenue': 0.0,
            'profit': 0.0,
            'debt': 0.0,
            
            # Production parameters
            'production_efficiency': production_efficiency,
            'labor_demand': 0.0,
            'energy_usage': 0.0,
            'goods_produced': 0.0,
            'goods_sold': 0.0,
            'price': 1.0,  # Initial normalized price
            
            # Strategy parameters
            'markup_rate': markup_rate,
            'labor_elasticity': self.labor_elasticity,
            'capital_elasticity': self.capital_elasticity,
            'energy_elasticity': self.energy_elasticity,
            
            # Tracking statistics
            'age': 0,
            'is_active': True
        }
    
    def update(self, state, model_state, model_config, key):
        """Update consumer goods firm behavior.
        
        The firm performs the following actions each step:
        1. Set production targets based on market demand
        2. Determine labor, capital and energy requirements
        3. Produce goods subject to constraints
        4. Set prices and sell goods
        5. Calculate profits and decide on investment
        
        Args:
            state: Current agent state
            model_state: Current model state including all environmental variables
            model_config: Model configuration parameters
            key: Random key for stochastic decisions
        
        Returns:
            Updated agent state
        """
        # Split random key for stochastic decisions
        subkey1, subkey2, subkey3, subkey4 = random.split(key, 4)
        
        # Using where to handle active/inactive firms in a JAX-compatible way
        # First compute the full updated state, then conditionally return it based on is_active
        
        # --- STEP 1: Market Assessment and Production Planning ---
        
        # Extract market conditions from environment
        market_demand = model_state['env'].get('consumer_goods_demand', 100.0)
        market_price = model_state['env'].get('consumer_goods_price', 1.0)
        wage_rate = model_state['env'].get('wage_rate', 1.0)
        energy_price = model_state['env'].get('energy_price', 1.0)
        capital_price = model_state['env'].get('capital_price', 1.0)
        interest_rate = model_state['env'].get('interest_rate', 0.05)
        
        # Check for environmental impacts
        climate_factor = model_state['env'].get('climate_impact', 1.0)
        pandemic_factor = model_state['env'].get('pandemic_impact', 1.0)
        
        # Firm's market share (simplified)
        firm_market_share = 0.01  # Assume equal market share for now
        
        # Target production based on market demand, adjusted for inventory
        target_production = market_demand * firm_market_share - state['inventory']
        target_production = jnp.maximum(0, target_production)  # Cannot be negative
        
        # --- STEP 2: Resource Planning ---
        
        # Determine optimal input mix with Cobb-Douglas production function
        # This is a simplified approach - in reality would solve cost minimization problem
        
        # Available cash for production
        available_funds = state['cash']
        
        # Calculate how much labor to hire based on target production
        # Derived from Cobb-Douglas with fixed capital
        base_labor_needed = (target_production / (state['production_efficiency'] * 
                            (state['capital_stock'] ** state['capital_elasticity']) * 
                            (1.0 ** state['energy_elasticity']))) ** (1 / state['labor_elasticity'])
        
        # Check if we can afford this labor
        max_affordable_labor = available_funds / wage_rate
        labor_demand = jnp.minimum(base_labor_needed, max_affordable_labor)
        
        # Calculate energy needed with similar approach
        base_energy_needed = (target_production / (state['production_efficiency'] * 
                             (state['capital_stock'] ** state['capital_elasticity']) * 
                             (labor_demand ** state['labor_elasticity']))) ** (1 / state['energy_elasticity'])
        
        # Remaining funds after labor
        remaining_funds = available_funds - (labor_demand * wage_rate)
        max_affordable_energy = remaining_funds / energy_price
        energy_usage = jnp.minimum(base_energy_needed, max_affordable_energy)
        
        # --- STEP 3: Production Process ---
        
        # Production function (Cobb-Douglas)
        production = state['production_efficiency'] * \
                     (labor_demand ** state['labor_elasticity']) * \
                     (state['capital_stock'] ** state['capital_elasticity']) * \
                     (energy_usage ** state['energy_elasticity'])
        
        # Apply environmental factors to production
        affected_production = production * climate_factor * pandemic_factor
        
        # Add randomness to production (technology/process variation)
        production_noise = random.normal(subkey1) * 0.05 + 1.0
        actual_production = affected_production * production_noise
        
        # Update inventory
        new_inventory = state['inventory'] + actual_production
        
        # --- STEP 4: Pricing and Sales ---
        
        # Calculate unit cost
        production_cost = (labor_demand * wage_rate) + (energy_usage * energy_price) + \
                         (state['capital_stock'] * capital_price * 0.05)  # Capital depreciation cost
        
        unit_cost = jnp.where(actual_production > 0, 
                             production_cost / actual_production, 
                             state['price'] / (1 + state['markup_rate']))
        
        # Set price with markup
        target_price = unit_cost * (1 + state['markup_rate'])
        
        # Adapt price gradually to market
        price_adjustment_speed = 0.3
        new_price = state['price'] * (1 - price_adjustment_speed) + target_price * price_adjustment_speed
        
        # Adjustment for market competition (lower price if inventory high)
        inventory_pressure = state['inventory'] / (actual_production + 0.1)
        price_discount = jnp.maximum(0, jnp.minimum(0.2, 0.05 * inventory_pressure))
        
        final_price = new_price * (1 - price_discount)
        
        # Determine sales based on market demand and price competitiveness
        price_elasticity = 1.2
        price_competitiveness = (market_price / final_price) ** price_elasticity
        
        # Random sales factor (reflects customer preferences, marketing, etc.)
        sales_randomness = random.uniform(subkey2) * 0.4 + 0.8
        
        # Calculate goods sold
        potential_sales = market_demand * firm_market_share * price_competitiveness * sales_randomness
        actual_sales = jnp.minimum(potential_sales, new_inventory)
        
        # Update inventory after sales
        final_inventory = new_inventory - actual_sales
        
        # --- STEP 5: Financial Calculations ---
        
        # Calculate revenue and costs
        revenue = actual_sales * final_price
        total_costs = production_cost + (state['debt'] * interest_rate)
        
        # Calculate profit
        profit = revenue - total_costs
        
        # Update cash position
        new_cash = state['cash'] + revenue - (labor_demand * wage_rate) - (energy_usage * energy_price)
        
        # Investment decision (simplified)
        investment_ratio = jnp.where(profit > 0, 0.3, 0.0)  # Invest 30% of profits if positive
        investment = profit * investment_ratio
        
        # Adjust investment to available cash
        actual_investment = jnp.minimum(investment, new_cash * 0.5)  # Don't use more than 50% of cash
        
        # Update capital stock (with depreciation)
        depreciation_rate = 0.05
        capital_purchases = actual_investment / capital_price
        new_capital = state['capital_stock'] * (1 - depreciation_rate) + capital_purchases
        
        # Final cash position after investment
        final_cash = new_cash - actual_investment
        
        # --- STEP 6: Update Firm Status ---
        
        # Check if firm is still viable
        is_viable = (final_cash > 0) & (new_capital > 0)
        
        # --- STEP 7: Return Updated State ---
        
        return {
            # Physical capital
            'capital_stock': new_capital,
            'production_capacity': new_capital * state['production_efficiency'],
            'inventory': final_inventory,
            
            # Financial state
            'cash': final_cash,
            'revenue': revenue,
            'profit': profit,
            'debt': state['debt'],  # Debt unchanged in this simple model
            
            # Production parameters
            'production_efficiency': state['production_efficiency'],
            'labor_demand': labor_demand,
            'energy_usage': energy_usage,
            'goods_produced': actual_production,
            'goods_sold': actual_sales,
            'price': final_price,
            
            # Strategy parameters remain the same
            'markup_rate': state['markup_rate'],
            'labor_elasticity': state['labor_elasticity'],
            'capital_elasticity': state['capital_elasticity'],
            'energy_elasticity': state['energy_elasticity'],
            
            # Tracking statistics
            'age': state['age'] + 1,
            'is_active': is_viable
        }

class CapitalGoodsFirm(AgentType):
    """Firms that produce capital goods using labor and energy."""
    
    def __init__(self, 
                initial_capital=1500.0, 
                initial_cash=800.0,
                production_efficiency=1.2,
                labor_elasticity=0.5,
                energy_elasticity=0.2,
                r_and_d_intensity=0.1,
                markup_rate=0.25):
        """Initialize capital goods firm parameters.
        
        Args:
            initial_capital: Starting capital stock
            initial_cash: Starting cash reserves
            production_efficiency: Total factor productivity
            labor_elasticity: Output elasticity of labor
            energy_elasticity: Output elasticity of energy
            r_and_d_intensity: Fraction of profits invested in R&D
            markup_rate: Profit margin added to production costs
        """
        self.initial_capital = initial_capital
        self.initial_cash = initial_cash
        self.production_efficiency = production_efficiency
        self.labor_elasticity = labor_elasticity
        self.energy_elasticity = energy_elasticity
        self.r_and_d_intensity = r_and_d_intensity
        self.markup_rate = markup_rate
    
    def init_state(self, model_config, key):
        """Initialize capital goods firm state.
        
        Creates the initial state for each firm, including:
        - Capital stock
        - Cash reserves
        - Production capacity
        - R&D investments
        - Technology level
        """
        # Add heterogeneity to firms
        subkey1, subkey2, subkey3 = random.split(key, 3)
        
        # Randomize initial capital (lognormal distribution)
        capital_factor = random.lognormal(subkey1, sigma=0.5)
        initial_capital = self.initial_capital * capital_factor
        
        # Randomize efficiency (normal distribution around mean)
        efficiency_factor = jnp.maximum(0.5, random.normal(subkey2) * 0.3 + 1.0)
        production_efficiency = self.production_efficiency * efficiency_factor
        
        # Randomize markup (beta distribution)
        markup_adj = random.beta(subkey3, a=5, b=2) * 0.3 + 0.1  # Range ~0.1-0.4
        markup_rate = self.markup_rate * markup_adj
        
        # Calculate initial production capacity based on capital
        production_capacity = production_efficiency * (initial_capital ** 0.5)
        
        # Initial technology level (represents complexity/quality of capital goods)
        technology_level = 1.0 * efficiency_factor
        
        return {
            # Physical capital
            'capital_stock': initial_capital,
            'production_capacity': production_capacity,
            'inventory': 0.0,
            
            # Financial state
            'cash': self.initial_cash,
            'revenue': 0.0,
            'profit': 0.0,
            'debt': 0.0,
            
            # Production parameters
            'production_efficiency': production_efficiency,
            'labor_demand': 0.0,
            'energy_usage': 0.0,
            'goods_produced': 0.0,
            'goods_sold': 0.0,
            'price': 2.0,  # Capital goods typically more expensive than consumer goods
            
            # Innovation parameters
            'technology_level': technology_level,
            'r_and_d_investment': 0.0,
            'r_and_d_success_prob': 0.0,
            
            # Strategy parameters
            'markup_rate': markup_rate,
            'labor_elasticity': self.labor_elasticity,
            'energy_elasticity': self.energy_elasticity,
            'r_and_d_intensity': self.r_and_d_intensity,
            
            # Tracking statistics
            'age': 0,
            'is_active': True
        }
    
    def update(self, state, model_state, model_config, key):
        """Update capital goods firm behavior.
        
        The firm performs the following actions each step:
        1. Set production targets based on market demand
        2. Determine labor and energy requirements
        3. Produce capital goods
        4. Set prices and sell to other firms
        5. Invest in R&D to improve technology
        6. Calculate profits and make investment decisions
        
        Args:
            state: Current agent state
            model_state: Current model state including all environmental variables
            model_config: Model configuration parameters
            key: Random key for stochastic decisions
        
        Returns:
            Updated agent state
        """
        # Split random key for stochastic decisions
        subkey1, subkey2, subkey3, subkey4 = random.split(key, 4)
        
        # --- STEP 1: Market Assessment and Production Planning ---
        
        # Extract market conditions from environment
        # Capital goods demand comes from consumer good firms and energy firms investing
        consumer_firms_investment = model_state['env'].get('consumer_firms_investment', 100.0)
        energy_firms_investment = model_state['env'].get('energy_firms_investment', 50.0)
        market_demand = consumer_firms_investment + energy_firms_investment
        
        market_price = model_state['env'].get('capital_goods_price', 2.0)
        wage_rate = model_state['env'].get('wage_rate', 1.0)
        energy_price = model_state['env'].get('energy_price', 1.0)
        interest_rate = model_state['env'].get('interest_rate', 0.05)
        
        # Check for environmental impacts
        climate_factor = model_state['env'].get('climate_impact', 1.0)
        pandemic_factor = model_state['env'].get('pandemic_impact', 1.0)
        
        # Firm's market share (simplified)
        total_capital_firms = model_state['env'].get('num_capital_firms', 20.0)
        firm_market_share = 1.0 / jnp.maximum(1.0, total_capital_firms)
        
        # Target production based on market demand, adjusted for inventory
        target_production = market_demand * firm_market_share - state['inventory']
        target_production = jnp.maximum(0, target_production)  # Cannot be negative
        
        # --- STEP 2: Resource Planning ---
        
        # Available cash for production
        available_funds = state['cash']
        
        # Calculate how much labor to hire based on target production
        # Two-factor production function (labor and energy)
        base_labor_needed = (target_production / (state['production_efficiency'] * 
                            (1.0 ** state['energy_elasticity']))) ** (1 / state['labor_elasticity'])
        
        # Check if we can afford this labor
        max_affordable_labor = available_funds / wage_rate
        labor_demand = jnp.minimum(base_labor_needed, max_affordable_labor)
        
        # Calculate energy needed
        base_energy_needed = (target_production / (state['production_efficiency'] * 
                             (labor_demand ** state['labor_elasticity']))) ** (1 / state['energy_elasticity'])
        
        # Remaining funds after labor
        remaining_funds = available_funds - (labor_demand * wage_rate)
        max_affordable_energy = remaining_funds / energy_price
        energy_usage = jnp.minimum(base_energy_needed, max_affordable_energy)
        
        # --- STEP 3: Production Process ---
        
        # Production function (labor and energy)
        production = state['production_efficiency'] * \
                     (labor_demand ** state['labor_elasticity']) * \
                     (energy_usage ** state['energy_elasticity']) * \
                     state['technology_level']  # Technology level boosts production
        
        # Apply environmental factors to production
        affected_production = production * climate_factor * pandemic_factor
        
        # Add randomness to production (technology/process variation)
        production_noise = random.normal(subkey1) * 0.05 + 1.0
        actual_production = affected_production * production_noise
        
        # Update inventory
        new_inventory = state['inventory'] + actual_production
        
        # --- STEP 4: Pricing and Sales ---
        
        # Calculate unit cost
        production_cost = (labor_demand * wage_rate) + (energy_usage * energy_price) + \
                         (state['capital_stock'] * market_price * 0.03)  # Capital maintenance cost
        
        unit_cost = jnp.where(actual_production > 0, 
                             production_cost / actual_production, 
                             state['price'] / (1 + state['markup_rate']))
        
        # Quality adjustment - higher technology firms can charge more
        quality_premium = state['technology_level'] - 1.0  # Premium for above-average technology
        
        # Set price with markup and quality premium
        target_price = unit_cost * (1 + state['markup_rate'] + jnp.maximum(0, quality_premium))
        
        # Adapt price gradually to market
        price_adjustment_speed = 0.3
        new_price = state['price'] * (1 - price_adjustment_speed) + target_price * price_adjustment_speed
        
        # Adjustment for market competition (lower price if inventory high)
        inventory_pressure = state['inventory'] / (actual_production + 0.1)
        price_discount = jnp.maximum(0, jnp.minimum(0.2, 0.05 * inventory_pressure))
        
        final_price = new_price * (1 - price_discount)
        
        # Determine sales based on market demand and price competitiveness
        price_elasticity = 1.0  # Capital goods typically less price elastic than consumer goods
        price_competitiveness = (market_price / final_price) ** price_elasticity
        
        # Quality factor - higher technology level increases demand
        quality_factor = 1.0 + jnp.maximum(0, (state['technology_level'] - 1.0) * 0.5)
        
        # Random sales factor (reflects firm relationships, reputation, etc.)
        sales_randomness = random.uniform(subkey2) * 0.3 + 0.85
        
        # Calculate goods sold
        potential_sales = market_demand * firm_market_share * price_competitiveness * quality_factor * sales_randomness
        actual_sales = jnp.minimum(potential_sales, new_inventory)
        
        # Update inventory after sales
        final_inventory = new_inventory - actual_sales
        
        # --- STEP 5: R&D and Innovation ---
        
        # Calculate revenue and profit before R&D
        revenue = actual_sales * final_price
        pre_rd_profit = revenue - production_cost - (state['debt'] * interest_rate)
        
        # R&D investment decision
        rd_investment_ratio = jnp.where(pre_rd_profit > 0, 
                                       state['r_and_d_intensity'], 
                                       0.0)  # Only invest in R&D when profitable
        
        rd_investment = pre_rd_profit * rd_investment_ratio
        
        # Success probability increases with investment but has diminishing returns
        rd_success_prob = 1.0 - jnp.exp(-0.5 * rd_investment / (100.0 + state['technology_level'] * 50.0))
        
        # Determine if R&D leads to technology improvement
        rd_success = random.uniform(subkey3) < rd_success_prob
        
        # Technology improvement magnitude depends on current level (diminishing returns)
        improvement_magnitude = 0.1 / jnp.sqrt(state['technology_level'])
        
        # New technology level
        new_technology = jnp.where(
            rd_success,
            state['technology_level'] * (1.0 + improvement_magnitude),
            state['technology_level']
        )
        
        # --- STEP 6: Financial Calculations ---
        
        # Final profit after R&D
        profit = pre_rd_profit - rd_investment
        
        # Update cash position
        new_cash = state['cash'] + revenue - production_cost - rd_investment
        
        # Investment decision for physical capital
        capital_investment_ratio = jnp.where(profit > 0, 0.2, 0.0)  # Invest 20% of profits in capital
        capital_investment = profit * capital_investment_ratio
        
        # Adjust investment to available cash
        actual_investment = jnp.minimum(capital_investment, new_cash * 0.4)
        
        # Update capital stock (with depreciation)
        depreciation_rate = 0.04
        capital_purchases = actual_investment / market_price
        new_capital = state['capital_stock'] * (1 - depreciation_rate) + capital_purchases
        
        # Final cash position after investment
        final_cash = new_cash - actual_investment
        
        # --- STEP 7: Update Firm Status ---
        
        # Check if firm is still viable
        is_viable = (final_cash > 0) & (new_capital > 0)
        
        # --- STEP 8: Return Updated State ---
        
        return {
            # Physical capital
            'capital_stock': new_capital,
            'production_capacity': new_capital * state['production_efficiency'],
            'inventory': final_inventory,
            
            # Financial state
            'cash': final_cash,
            'revenue': revenue,
            'profit': profit,
            'debt': state['debt'],  # Debt unchanged in this simple model
            
            # Production parameters
            'production_efficiency': state['production_efficiency'],
            'labor_demand': labor_demand,
            'energy_usage': energy_usage,
            'goods_produced': actual_production,
            'goods_sold': actual_sales,
            'price': final_price,
            
            # Innovation parameters
            'technology_level': new_technology,
            'r_and_d_investment': rd_investment,
            'r_and_d_success_prob': rd_success_prob,
            
            # Strategy parameters remain the same
            'markup_rate': state['markup_rate'],
            'labor_elasticity': state['labor_elasticity'],
            'energy_elasticity': state['energy_elasticity'],
            'r_and_d_intensity': state['r_and_d_intensity'],
            
            # Tracking statistics
            'age': state['age'] + 1,
            'is_active': is_viable
        }

class EnergyFirm(AgentType):
    """Firms that produce energy using capital and possibly labor."""
    
    def __init__(self, 
                initial_capital=2000.0, 
                initial_cash=1000.0,
                production_efficiency=1.5,
                capital_elasticity=0.7,
                labor_elasticity=0.3,
                renewable_fraction=0.2,
                markup_rate=0.15):
        """Initialize energy firm parameters.
        
        Args:
            initial_capital: Starting capital stock
            initial_cash: Starting cash reserves
            production_efficiency: Total factor productivity
            capital_elasticity: Output elasticity of capital
            labor_elasticity: Output elasticity of labor
            renewable_fraction: Portion of energy from renewable sources
            markup_rate: Profit margin added to production costs
        """
        self.initial_capital = initial_capital
        self.initial_cash = initial_cash
        self.production_efficiency = production_efficiency
        self.capital_elasticity = capital_elasticity
        self.labor_elasticity = labor_elasticity
        self.renewable_fraction = renewable_fraction
        self.markup_rate = markup_rate
    
    def init_state(self, model_config, key):
        """Initialize energy firm state.
        
        Creates the initial state for each firm, including:
        - Capital stock
        - Cash reserves
        - Production capacity
        - Energy production mix (renewable vs. non-renewable)
        """
        # Add heterogeneity to firms
        subkey1, subkey2, subkey3, subkey4 = random.split(key, 4)
        
        # Randomize initial capital (lognormal distribution)
        capital_factor = random.lognormal(subkey1, sigma=0.6)
        initial_capital = self.initial_capital * capital_factor
        
        # Randomize efficiency (normal distribution around mean)
        efficiency_factor = jnp.maximum(0.5, random.normal(subkey2) * 0.3 + 1.0)
        production_efficiency = self.production_efficiency * efficiency_factor
        
        # Randomize markup (beta distribution)
        markup_adj = random.beta(subkey3, a=5, b=3) * 0.2 + 0.1  # Range ~0.1-0.3
        markup_rate = self.markup_rate * markup_adj
        
        # Randomize renewable fraction (beta distribution)
        renewable_adj = random.beta(subkey4, a=2, b=5) * 2.0  # More variance
        renewable_fraction = jnp.minimum(1.0, self.renewable_fraction * renewable_adj)
        
        # Calculate initial production capacity based on capital
        production_capacity = production_efficiency * (initial_capital ** self.capital_elasticity)
        
        return {
            # Physical capital
            'capital_stock': initial_capital,
            'production_capacity': production_capacity,
            'energy_storage': 0.0,  # Energy can be stored to some extent
            
            # Financial state
            'cash': self.initial_cash,
            'revenue': 0.0,
            'profit': 0.0,
            'debt': 0.0,
            
            # Production parameters
            'production_efficiency': production_efficiency,
            'labor_demand': 0.0,
            'carbon_emissions': 0.0,
            'energy_produced': 0.0,
            'energy_sold': 0.0,
            'price': 1.0,  # Initial normalized price
            
            # Energy mix parameters
            'renewable_fraction': renewable_fraction,
            'fossil_fuel_usage': 0.0,
            'renewable_capacity': initial_capital * renewable_fraction,
            
            # Strategy parameters
            'markup_rate': markup_rate,
            'capital_elasticity': self.capital_elasticity,
            'labor_elasticity': self.labor_elasticity,
            
            # Tracking statistics
            'age': 0,
            'is_active': True
        }
    
    def update(self, state, model_state, model_config, key):
        """Update energy firm behavior.
        
        The firm performs the following actions each step:
        1. Set production targets based on market demand
        2. Determine input mix (capital, labor, fossil fuels)
        3. Produce energy
        4. Set prices and sell energy
        5. Calculate carbon emissions
        6. Make investment decisions in renewable vs. non-renewable capacity
        
        Args:
            state: Current agent state
            model_state: Current model state including all environmental variables
            model_config: Model configuration parameters
            key: Random key for stochastic decisions
        
        Returns:
            Updated agent state
        """
        # Split random key for stochastic decisions
        subkey1, subkey2, subkey3, subkey4 = random.split(key, 4)
        
        # --- STEP 1: Market Assessment and Production Planning ---
        
        # Extract market conditions from environment
        # Energy demand comes from household consumption, consumer firms and capital goods firms
        household_energy_demand = model_state['env'].get('household_energy_demand', 50.0)
        consumer_firms_energy_demand = model_state['env'].get('consumer_firms_energy_usage', 100.0)
        capital_firms_energy_demand = model_state['env'].get('capital_firms_energy_usage', 50.0)
        total_energy_demand = household_energy_demand + consumer_firms_energy_demand + capital_firms_energy_demand
        
        market_price = model_state['env'].get('energy_price', 1.0)
        wage_rate = model_state['env'].get('wage_rate', 1.0)
        fossil_fuel_price = model_state['env'].get('fossil_fuel_price', 0.8)
        capital_price = model_state['env'].get('capital_goods_price', 2.0)
        interest_rate = model_state['env'].get('interest_rate', 0.05)
        
        # Environmental factors and policy
        climate_policy_strength = model_state['env'].get('climate_policy_strength', 0.0)
        carbon_price = model_state['env'].get('carbon_price', 0.0)
        renewable_subsidy = model_state['env'].get('renewable_subsidy', 0.0)
        climate_factor = model_state['env'].get('climate_impact', 1.0)
        pandemic_factor = model_state['env'].get('pandemic_impact', 1.0)
        
        # Firm's market share (simplified)
        total_energy_firms = model_state['env'].get('num_energy_firms', 10.0)
        firm_market_share = 1.0 / jnp.maximum(1.0, total_energy_firms)
        
        # Target production based on market demand, adjusted for storage
        target_production = total_energy_demand * firm_market_share - state['energy_storage']
        target_production = jnp.maximum(0, target_production)  # Cannot be negative
        
        # --- STEP 2: Production Planning ---
        
        # Available cash for production
        available_funds = state['cash']
        
        # Calculate production from renewable sources (dependent on capital)
        renewable_production = state['renewable_capacity'] * state['production_efficiency'] * \
                              (random.normal(subkey1) * 0.2 + 0.9)  # Renewable output varies (sun/wind)
        
        # If renewable production is less than target, use fossil fuels and labor to make up the difference
        remaining_production_needed = jnp.maximum(0, target_production - renewable_production)
        
        # Calculate how much labor to hire based on remaining need
        base_labor_needed = jnp.where(
            remaining_production_needed > 0,
            (remaining_production_needed / (
                state['production_efficiency'] * 
                ((1.0 - state['renewable_fraction']) * state['capital_stock']) ** state['capital_elasticity']
            )) ** (1 / state['labor_elasticity']),
            0.0
        )
        
        # Check if we can afford this labor
        max_affordable_labor = available_funds / wage_rate
        labor_demand = jnp.minimum(base_labor_needed, max_affordable_labor)
        
        # Calculate fossil fuel needed (based on non-renewable fraction of production)
        non_renewable_production = jnp.where(
            labor_demand > 0,
            state['production_efficiency'] * 
            ((1.0 - state['renewable_fraction']) * state['capital_stock']) ** state['capital_elasticity'] *
            labor_demand ** state['labor_elasticity'],
            0.0
        )
        
        # Amount of fossil fuel needed depends on efficiency
        fossil_efficiency = 2.0  # Energy units per fossil fuel unit
        fossil_fuel_needed = non_renewable_production / fossil_efficiency
        
        # Check if we can afford fossil fuels
        remaining_funds = available_funds - (labor_demand * wage_rate)
        max_affordable_fossil = remaining_funds / fossil_fuel_price
        fossil_fuel_usage = jnp.minimum(fossil_fuel_needed, max_affordable_fossil)
        
        # --- STEP 3: Energy Production ---
        
        # Production from non-renewable sources
        actual_non_renewable = jnp.where(
            fossil_fuel_usage > 0,
            fossil_fuel_usage * fossil_efficiency,
            0.0
        )
        
        # Total energy produced
        total_production = renewable_production + actual_non_renewable
        
        # Apply climate impacts (can affect both renewable and non-renewable)
        climate_impact_factor = 1.0 - jnp.maximum(0, (climate_factor - 1.0) * 0.2)
        pandemic_impact_factor = pandemic_factor  # Less impact on energy than other sectors
        
        # Environmental factors affect production
        affected_production = total_production * climate_impact_factor * pandemic_impact_factor
        
        # Add randomness to production (technical issues, maintenance, etc.)
        production_noise = random.normal(subkey2) * 0.05 + 1.0
        actual_production = affected_production * production_noise
        
        # Calculate carbon emissions (only from non-renewable production)
        carbon_emission_rate = 0.5  # Carbon units per non-renewable energy unit
        carbon_emissions = actual_non_renewable * carbon_emission_rate
        
        # Update energy storage
        new_storage = state['energy_storage'] + actual_production
        
        # --- STEP 4: Pricing and Sales ---
        
        # Calculate production costs
        labor_cost = labor_demand * wage_rate
        fossil_fuel_cost = fossil_fuel_usage * fossil_fuel_price
        capital_maintenance = state['capital_stock'] * 0.02  # 2% maintenance cost
        carbon_tax = carbon_emissions * carbon_price
        
        total_cost = labor_cost + fossil_fuel_cost + capital_maintenance + carbon_tax - \
                    (renewable_production * renewable_subsidy)  # Subsidy reduces costs
        
        # Calculate unit cost
        unit_cost = jnp.where(actual_production > 0, 
                             total_cost / actual_production, 
                             state['price'] / (1 + state['markup_rate']))
        
        # Set price with markup
        target_price = unit_cost * (1 + state['markup_rate'])
        
        # Adapt price gradually to market
        price_adjustment_speed = 0.2
        new_price = state['price'] * (1 - price_adjustment_speed) + target_price * price_adjustment_speed
        
        # Adjustment for market competition (lower price if storage high)
        storage_pressure = state['energy_storage'] / (actual_production + 0.1)
        price_discount = jnp.maximum(0, jnp.minimum(0.15, 0.05 * storage_pressure))
        
        final_price = new_price * (1 - price_discount)
        
        # Determine sales based on market demand and price competitiveness
        price_elasticity = 0.8  # Energy demand is relatively inelastic
        price_competitiveness = (market_price / final_price) ** price_elasticity
        
        # Green energy premium - higher renewable fraction can command slightly higher prices
        green_premium = state['renewable_fraction'] * climate_policy_strength * 0.1
        
        # Random sales factor (reflects weather, seasonal factors, etc.)
        sales_randomness = random.uniform(subkey3) * 0.2 + 0.9
        
        # Calculate energy sold
        potential_sales = total_energy_demand * firm_market_share * price_competitiveness * \
                         (1.0 + green_premium) * sales_randomness
        actual_sales = jnp.minimum(potential_sales, new_storage)
        
        # Update storage after sales
        final_storage = new_storage - actual_sales
        
        # --- STEP 5: Financial Calculations ---
        
        # Calculate revenue and costs
        revenue = actual_sales * final_price
        total_costs = total_cost + (state['debt'] * interest_rate)
        
        # Calculate profit
        profit = revenue - total_costs
        
        # Update cash position
        new_cash = state['cash'] + revenue - total_costs
        
        # --- STEP 6: Investment Decisions ---
        
        # Investment decision based on profit
        investment_ratio = jnp.where(profit > 0, 0.3, 0.0)  # Invest 30% of profits if positive
        investment = profit * investment_ratio
        
        # Adjust investment to available cash
        actual_investment = jnp.minimum(investment, new_cash * 0.5)
        
        # Allocate investment between renewable and non-renewable
        # Higher climate policy and carbon price increase renewable investment
        renewable_investment_share = jnp.minimum(
            0.9,  # Cap at 90%
            state['renewable_fraction'] + climate_policy_strength * 0.2 + carbon_price * 0.1
        )
        
        # Investment in renewable capacity
        renewable_investment = actual_investment * renewable_investment_share
        
        # Investment in general capital (non-renewable)
        general_investment = actual_investment * (1.0 - renewable_investment_share)
        
        # Update capital stock (with depreciation)
        depreciation_rate = 0.03
        renewable_purchases = renewable_investment / capital_price
        general_purchases = general_investment / capital_price
        
        new_renewable_capacity = state['renewable_capacity'] * (1 - depreciation_rate) + renewable_purchases
        new_capital = state['capital_stock'] * (1 - depreciation_rate) + general_purchases + renewable_purchases
        
        # Update renewable fraction based on new capacity
        new_renewable_fraction = jnp.where(
            new_capital > 0,
            new_renewable_capacity / new_capital,
            state['renewable_fraction']
        )
        
        # Final cash position after investment
        final_cash = new_cash - actual_investment
        
        # --- STEP 7: Update Firm Status ---
        
        # Check if firm is still viable
        is_viable = (final_cash > 0) & (new_capital > 0)
        
        # --- STEP 8: Return Updated State ---
        
        return {
            # Physical capital
            'capital_stock': new_capital,
            'production_capacity': new_capital * state['production_efficiency'],
            'energy_storage': final_storage,
            
            # Financial state
            'cash': final_cash,
            'revenue': revenue,
            'profit': profit,
            'debt': state['debt'],  # Debt unchanged in this simple model
            
            # Production parameters
            'production_efficiency': state['production_efficiency'],
            'labor_demand': labor_demand,
            'carbon_emissions': carbon_emissions,
            'energy_produced': actual_production,
            'energy_sold': actual_sales,
            'price': final_price,
            
            # Energy mix parameters
            'renewable_fraction': new_renewable_fraction,
            'fossil_fuel_usage': fossil_fuel_usage,
            'renewable_capacity': new_renewable_capacity,
            
            # Strategy parameters remain the same
            'markup_rate': state['markup_rate'],
            'capital_elasticity': state['capital_elasticity'],
            'labor_elasticity': state['labor_elasticity'],
            
            # Tracking statistics
            'age': state['age'] + 1,
            'is_active': is_viable
        }

# === Government Agent ===
class Government(AgentType):
    """Government collects taxes, issues bonds, and provides transfers."""
    
    def __init__(self):
        """Initialize government parameters."""
        pass
    
    def init_state(self, model_config, key):
        """Initialize government state."""
        # To be implemented
        pass
    
    def update(self, state, model_state, model_config, key):
        """Update government behavior: collect taxes, issue bonds, transfers."""
        # To be implemented
        pass

# === Bank Agent ===
class Bank(AgentType):
    """Banks manage savings and provide credit."""
    
    def __init__(self):
        """Initialize bank parameters."""
        pass
    
    def init_state(self, model_config, key):
        """Initialize bank state."""
        # To be implemented
        pass
    
    def update(self, state, model_state, model_config, key):
        """Update bank behavior: manage deposits, provide loans."""
        # To be implemented
        pass

# ------------- MODEL COMPONENTS -------------

# Climate Module
def climate_impact(env_state, agent_states, params, key):
    """Calculate climate impacts on the economy.
    
    This module simulates climate-related disruptions to production,
    energy prices, and resource availability.
    
    Args:
        env_state: Current environmental state
        agent_states: States of all agent collections
        params: Model parameters
        key: Random key for stochastic effects
        
    Returns:
        Climate impact factors to be applied to the economy
    """
    # Extract relevant parameters
    climate_policy_strength = params.get('climate_policy_strength', 0.5)
    carbon_price = params.get('carbon_price', 0.0)
    renewable_subsidy = params.get('renewable_subsidy', 0.0)
    current_time = env_state.get('time_step', 0)
    
    # Split random key for stochastic events
    subkey1, subkey2 = random.split(key)
    
    # Calculate baseline climate trend (warming over time)
    baseline_trend = 0.001 * current_time  # Gradual warming
    
    # Stochastic extreme weather events (floods, droughts, storms)
    # Higher probability as baseline trend increases
    extreme_event_probability = 0.01 + baseline_trend * 0.1
    extreme_event_occurs = random.uniform(subkey1) < extreme_event_probability
    
    extreme_event_magnitude = jnp.where(
        extreme_event_occurs,
        random.uniform(subkey2, minval=0.1, maxval=0.3),
        0.0
    )
    
    # Calculate impacts based on climate policy strength
    mitigation_effect = climate_policy_strength * 0.5  # Stronger policy reduces impacts
    
    # Energy sector impact - affects energy prices and production
    energy_impact = jnp.clip(1.0 + baseline_trend * 0.5 + extreme_event_magnitude - mitigation_effect, 0.8, 1.5)
    
    # Agricultural impact - affects consumer goods production
    agricultural_impact = jnp.clip(1.0 - baseline_trend * 0.8 - extreme_event_magnitude + mitigation_effect, 0.6, 1.0)
    
    # Infrastructure impact - affects capital goods production
    infrastructure_impact = jnp.clip(1.0 - extreme_event_magnitude * 0.5 + mitigation_effect * 0.3, 0.9, 1.0)
    
    # Return dictionary of impact factors
    return {
        'climate_trend': baseline_trend,
        'extreme_event_magnitude': extreme_event_magnitude,
        'energy_price_factor': energy_impact,
        'agricultural_productivity_factor': agricultural_impact,
        'infrastructure_productivity_factor': infrastructure_impact,
        'overall_climate_impact': jnp.mean(jnp.array([energy_impact, agricultural_impact, infrastructure_impact]))
    }

# Pandemic Module
def pandemic_impact(env_state, agent_states, params, key):
    """Calculate pandemic impacts on the economy.
    
    This module simulates pandemic-related disruptions to labor supply,
    consumption patterns, and overall economic activity.
    
    Args:
        env_state: Current environmental state
        agent_states: States of all agent collections
        params: Model parameters
        key: Random key for stochastic effects
        
    Returns:
        Pandemic impact factors to be applied to the economy
    """
    # Extract relevant parameters
    pandemic_active = params.get('pandemic_active', False)
    pandemic_severity = params.get('pandemic_severity', 0.0)
    healthcare_capacity = params.get('healthcare_capacity', 1.0)
    social_distancing = params.get('social_distancing', 0.0)
    vaccination_rate = params.get('vaccination_rate', 0.0)
    
    # Current pandemic state
    current_infected = env_state.get('pandemic_infected_rate', 0.0)
    
    # If pandemic not active, return neutral impacts
    if not pandemic_active:
        return {
            'pandemic_infected_rate': 0.0,
            'labor_supply_impact': 1.0,
            'consumption_impact': 1.0,
            'service_sector_impact': 1.0,
            'manufacturing_impact': 1.0,
            'healthcare_burden': 0.0,
            'overall_pandemic_impact': 1.0
        }
    
    # Split key for stochastic components
    subkey1, subkey2 = random.split(key)
    
    # Basic SIR-type pandemic model
    # Calculate new infection rate based on current state and mitigation measures
    infection_factor = pandemic_severity * (1.0 - social_distancing * 0.8) * (1.0 - vaccination_rate * 0.9)
    
    # Stochastic component to infection spread
    infection_noise = random.normal(subkey1) * 0.1 + 1.0
    
    # New infected rate follows logistic growth pattern initially, then declines
    theoretical_new_rate = infection_factor * current_infected * (1.0 - current_infected) * infection_noise
    
    # Recovery rate depends on healthcare capacity
    recovery_rate = 0.1 * healthcare_capacity
    
    # Update infected rate
    new_infected_rate = current_infected + theoretical_new_rate - recovery_rate * current_infected
    new_infected_rate = jnp.clip(new_infected_rate, 0.0, 0.5)  # Cap at 50% infected
    
    # Calculate economic impacts
    
    # Labor supply (reduced by infections and social distancing)
    labor_impact = 1.0 - new_infected_rate * 0.8 - social_distancing * 0.2 * (1.0 - vaccination_rate)
    
    # Consumption (reduced by infections, social distancing, and economic uncertainty)
    consumption_uncertainty = random.uniform(subkey2, minval=0.0, maxval=0.2) * pandemic_severity
    consumption_impact = 1.0 - new_infected_rate * 0.5 - social_distancing * 0.3 - consumption_uncertainty
    
    # Sector-specific impacts
    # Services hit harder by social distancing
    service_impact = 1.0 - social_distancing * 0.5 - new_infected_rate * 0.3
    
    # Manufacturing somewhat resilient but affected by labor shortages
    manufacturing_impact = 1.0 - new_infected_rate * 0.2 - social_distancing * 0.1
    
    # Healthcare burden
    healthcare_burden = new_infected_rate / healthcare_capacity
    
    # Overall economic impact is a weighted average
    overall_impact = 0.4 * labor_impact + 0.4 * consumption_impact + 0.1 * service_impact + 0.1 * manufacturing_impact
    
    return {
        'pandemic_infected_rate': new_infected_rate,
        'labor_supply_impact': labor_impact,
        'consumption_impact': consumption_impact,
        'service_sector_impact': service_impact,
        'manufacturing_impact': manufacturing_impact,
        'healthcare_burden': healthcare_burden,
        'overall_pandemic_impact': overall_impact
    }

# Environment update function
def update_environment(env_state, agent_states, params, key):
    """Update global economic environment state based on agent actions.
    
    This function aggregates the collective actions of all agents to update
    global economic variables like prices, interest rates, and market conditions.
    
    Args:
        env_state: Current environmental state
        agent_states: States of all agent collections
        params: Model parameters
        key: Random key for stochastic components
        
    Returns:
        Updated environmental state
    """
    # Split random key
    subkey1, subkey2, subkey3, subkey4, subkey5 = random.split(key, 5)
    
    # --- STEP 1: Extract and process household data ---
    
    # Get household states
    household_states = agent_states.get('households', {})
    
    if not household_states:  # No households
        return env_state
    
    # Calculate aggregate household metrics
    total_labor_supply = jnp.sum(household_states.get('labor_supply', jnp.array([0.0])))
    total_consumption = jnp.sum(household_states.get('consumption', jnp.array([0.0])))
    total_savings = jnp.sum(household_states.get('savings', jnp.array([0.0])))
    total_bank_deposits = jnp.sum(household_states.get('bank_deposits', jnp.array([0.0])))
    total_income = jnp.sum(household_states.get('income', jnp.array([0.0])))
    employment_rate = jnp.mean(household_states.get('employed', jnp.array([0.0])).astype(float))
    avg_utility = jnp.mean(household_states.get('utility', jnp.array([0.0])))
    
    # --- STEP 2: Extract and process firm data ---
    
    # Get different firm states
    consumer_firm_states = agent_states.get('consumer_firms', {})
    capital_firm_states = agent_states.get('capital_firms', {})
    energy_firm_states = agent_states.get('energy_firms', {})
    
    # Consumer goods market
    if consumer_firm_states:
        consumer_goods_produced = jnp.sum(consumer_firm_states.get('goods_produced', jnp.array([0.0])))
        consumer_goods_sold = jnp.sum(consumer_firm_states.get('goods_sold', jnp.array([0.0])))
        consumer_goods_inventory = jnp.sum(consumer_firm_states.get('inventory', jnp.array([0.0])))
        consumer_goods_price = jnp.mean(consumer_firm_states.get('price', jnp.array([1.0])))
        consumer_firms_labor_demand = jnp.sum(consumer_firm_states.get('labor_demand', jnp.array([0.0])))
        consumer_firms_energy_usage = jnp.sum(consumer_firm_states.get('energy_usage', jnp.array([0.0])))
        consumer_firms_profit = jnp.sum(consumer_firm_states.get('profit', jnp.array([0.0])))
        
        # Consumer firms investment (for capital goods demand)
        # Base investment on profits and a constant capital replacement rate
        capital_replacement = jnp.sum(consumer_firm_states.get('capital_stock', jnp.array([0.0]))) * 0.05
        capital_expansion = jnp.maximum(0.0, consumer_firms_profit) * 0.3
        consumer_firms_investment = capital_replacement + capital_expansion
    else:
        consumer_goods_produced = 0.0
        consumer_goods_sold = 0.0
        consumer_goods_inventory = 0.0
        consumer_goods_price = 1.0
        consumer_firms_labor_demand = 0.0
        consumer_firms_energy_usage = 0.0
        consumer_firms_investment = 0.0
    
    # Capital goods market
    if capital_firm_states:
        capital_goods_produced = jnp.sum(capital_firm_states.get('goods_produced', jnp.array([0.0])))
        capital_goods_sold = jnp.sum(capital_firm_states.get('goods_sold', jnp.array([0.0])))
        capital_goods_inventory = jnp.sum(capital_firm_states.get('inventory', jnp.array([0.0])))
        capital_goods_price = jnp.mean(capital_firm_states.get('price', jnp.array([2.0])))
        capital_firms_labor_demand = jnp.sum(capital_firm_states.get('labor_demand', jnp.array([0.0])))
        capital_firms_energy_usage = jnp.sum(capital_firm_states.get('energy_usage', jnp.array([0.0])))
        avg_technology_level = jnp.mean(capital_firm_states.get('technology_level', jnp.array([1.0])))
    else:
        capital_goods_produced = 0.0
        capital_goods_sold = 0.0
        capital_goods_inventory = 0.0
        capital_goods_price = 2.0
        capital_firms_labor_demand = 0.0
        capital_firms_energy_usage = 0.0
        avg_technology_level = 1.0
    
    # Energy market
    if energy_firm_states:
        energy_produced = jnp.sum(energy_firm_states.get('energy_produced', jnp.array([0.0])))
        energy_sold = jnp.sum(energy_firm_states.get('energy_sold', jnp.array([0.0])))
        energy_storage = jnp.sum(energy_firm_states.get('energy_storage', jnp.array([0.0])))
        energy_price = jnp.mean(energy_firm_states.get('price', jnp.array([1.0])))
        energy_firms_labor_demand = jnp.sum(energy_firm_states.get('labor_demand', jnp.array([0.0])))
    else:
        energy_produced = 100.0  # Base energy production if no energy firms
        energy_price = 1.0
        energy_firms_labor_demand = 0.0
    
    # --- STEP 3: Extract government and bank data ---
    
    # Government data
    govt_states = agent_states.get('government', {})
    if govt_states:
        tax_revenue = jnp.sum(govt_states.get('tax_revenue', jnp.array([0.0])))
        govt_spending = jnp.sum(govt_states.get('spending', jnp.array([0.0])))
        public_debt = jnp.sum(govt_states.get('debt', jnp.array([0.0])))
    else:
        # Default values if no government
        tax_revenue = 0.0
        govt_spending = 0.0
        public_debt = 0.0
    
    # Bank data
    bank_states = agent_states.get('banks', {})
    if bank_states:
        total_deposits = jnp.sum(bank_states.get('deposits', jnp.array([0.0])))
        total_loans = jnp.sum(bank_states.get('loans', jnp.array([0.0])))
        avg_interest_rate = jnp.mean(bank_states.get('interest_rate', jnp.array([0.05])))
    else:
        # Default values if no banks
        total_deposits = total_bank_deposits  # Use household deposits
        total_loans = 0.0
        avg_interest_rate = 0.05
    
    # --- STEP 4: Market clearing and price adjustments ---
    
    # Extract current prices from environment
    old_wage_rate = env_state.get('wage_rate', 1.0)
    old_price_level = env_state.get('price_level', 1.0)
    old_interest_rate = env_state.get('interest_rate', 0.05)
    
    # Labor market clearing
    total_labor_demand = consumer_firms_labor_demand + capital_firms_labor_demand + energy_firms_labor_demand
    
    # Labor market balance - affects wage rate
    # Add a small constant to prevent division by zero
    labor_market_tightness = jnp.where(total_labor_supply > 1e-6, 
                                      total_labor_demand / (total_labor_supply + 1e-6), 
                                      1.0)
    
    # Adjust wage based on labor market pressure
    wage_adjustment_factor = 0.2
    wage_pressure = (labor_market_tightness - 1.0) * wage_adjustment_factor
    
    # Add small random component to wage adjustment
    wage_noise = random.normal(subkey1) * 0.01
    
    # Calculate new wage rate with limits on change
    wage_change = jnp.clip(wage_pressure + wage_noise, -0.05, 0.05)
    new_wage_rate = jnp.maximum(0.1, old_wage_rate * (1.0 + wage_change))  # Ensure wage rate doesn't go below 0.1
    
    # Goods market - calculate overall price level
    # Weighted average of consumer, capital, and energy prices
    price_weights = jnp.array([0.6, 0.3, 0.1])  # Consumer, capital, energy
    price_indices = jnp.array([consumer_goods_price, capital_goods_price, energy_price])
    new_price_level = jnp.sum(price_weights * price_indices)
    new_price_level = jnp.maximum(0.1, new_price_level)  # Ensure price level doesn't go below 0.1
    
    # Credit market - interest rate adjustment
    # Based on central bank policy (Taylor rule-inspired)
    inflation_rate = (new_price_level / jnp.maximum(0.1, old_price_level)) - 1.0
    unemployment_rate = 1.0 - employment_rate
    
    # Simple Taylor rule for interest rate
    target_inflation = 0.02  # 2% inflation target
    natural_rate = 0.02      # Natural interest rate
    
    inflation_gap = inflation_rate - target_inflation
    output_gap = -0.5 * (unemployment_rate - 0.05)  # Using unemployment as proxy
    
    taylor_rate = natural_rate + 1.5 * inflation_gap + 0.5 * output_gap
    
    # Gradual adjustment of interest rate with bounds
    interest_adjustment_factor = 0.3
    interest_rate_change = (taylor_rate - old_interest_rate) * interest_adjustment_factor
    
    # Add small random component
    interest_noise = random.normal(subkey2) * 0.005
    
    # Bounded change
    interest_rate_change = jnp.clip(interest_rate_change + interest_noise, -0.02, 0.02)
    new_interest_rate = jnp.maximum(0.01, old_interest_rate + interest_rate_change)
    
    # --- STEP 5: Calculate economic indicators ---
    
    # GDP (simplified) = Consumption + Investment + Government Spending
    # Using production as proxy for different components
    gdp = consumer_goods_produced * consumer_goods_price + \
          capital_goods_produced * capital_goods_price + \
          govt_spending
    
    # Set a minimum GDP to prevent NaN in growth calculations
    gdp = jnp.maximum(0.1, gdp)
    
    # Previous GDP for growth calculation
    prev_gdp = jnp.maximum(0.1, env_state.get('gdp', gdp))  # Ensure prev_gdp is at least 0.1
    gdp_growth = (gdp / prev_gdp) - 1.0
    
    # --- STEP 6: Apply external module impacts ---
    
    # Apply Climate Module if enabled
    if params.get('enable_climate_module', False):
        climate_effects = climate_impact(env_state, agent_states, params, subkey3)
        climate_impact_factor = climate_effects['overall_climate_impact']
    else:
        climate_impact_factor = 1.0
        climate_effects = {'overall_climate_impact': 1.0}
    
    # Apply Pandemic Module if enabled
    if params.get('enable_pandemic_module', False):
        pandemic_effects = pandemic_impact(env_state, agent_states, params, subkey4)
        pandemic_impact_factor = pandemic_effects['overall_pandemic_impact']
    else:
        pandemic_impact_factor = 1.0
        pandemic_effects = {'overall_pandemic_impact': 1.0}
    
    # --- STEP 7: Combine everything into new environment state ---
    
    # Calculate income per capita safely
    household_count = household_states.get('position', jnp.array([0])).shape[0]
    income_per_capita = total_income / jnp.maximum(1, household_count)  # Ensure household_count is at least 1
    
    new_env_state = {
        # Time tracking
        'time_step': env_state.get('time_step', 0) + 1,
        
        # Labor market
        'wage_rate': new_wage_rate,
        'total_labor_supply': total_labor_supply,
        'total_labor_demand': total_labor_demand,
        'employment_rate': employment_rate,
        'unemployment_rate': 1.0 - employment_rate,
        'job_market_condition': labor_market_tightness,
        
        # Goods markets
        'price_level': new_price_level,
        'inflation_rate': inflation_rate,
        'consumer_goods_price': consumer_goods_price,
        'capital_goods_price': capital_goods_price,
        'energy_price': energy_price,
        'consumer_goods_supply': consumer_goods_produced,
        'consumer_goods_demand': total_consumption,
        'consumer_goods_inventory': consumer_goods_inventory,
        'goods_availability': jnp.minimum(1.0, consumer_goods_produced / jnp.maximum(1e-5, total_consumption)),
        
        # Financial markets
        'interest_rate': new_interest_rate,
        'total_savings': total_savings,
        'total_deposits': total_deposits,
        'total_loans': total_loans,
        
        # Fiscal variables
        'tax_revenue': tax_revenue,
        'govt_spending': govt_spending,
        'public_debt': public_debt,
        'debt_to_gdp': public_debt / jnp.maximum(0.1, gdp),  # Ensure gdp is at least 0.1
        
        # Economic indicators
        'gdp': gdp,
        'gdp_growth': gdp_growth,
        'avg_utility': avg_utility,
        'total_income': total_income,
        'income_per_capita': income_per_capita,
        
        # External impacts
        'climate_impact': climate_impact_factor,
        'pandemic_impact': pandemic_impact_factor,
        
        # Detailed external module states
        **climate_effects,
        **pandemic_effects,
        
        # Keep any other existing state variables not explicitly updated
        **{k: v for k, v in env_state.items() 
           if k not in ['time_step', 'wage_rate', 'price_level', 'interest_rate',
                        'gdp', 'gdp_growth', 'total_labor_supply', 'total_labor_demand',
                        'employment_rate', 'unemployment_rate', 'climate_impact', 'pandemic_impact']}
    }
    
    return new_env_state

# Metrics calculation
def compute_metrics(env_state, agent_states, params):
    """Compute economic metrics from model state.
    
    Calculates key economic indicators and statistics from the
    current model state for tracking and analysis.
    
    Args:
        env_state: Current environmental state
        agent_states: States of all agent collections
        params: Model parameters
        
    Returns:
        Dictionary of metrics
    """
    # --- Economic output metrics ---
    gdp = env_state.get('gdp', 0.1)  # Minimum GDP of 0.1 to prevent NaN
    gdp_growth = env_state.get('gdp_growth', 0.0)
    price_level = env_state.get('price_level', 1.0)
    inflation_rate = env_state.get('inflation_rate', 0.0)
    
    # --- Labor market metrics ---
    unemployment_rate = env_state.get('unemployment_rate', 0.0)
    employment_rate = env_state.get('employment_rate', 1.0)
    wage_rate = env_state.get('wage_rate', 1.0)
    labor_market_tightness = env_state.get('job_market_condition', 1.0)
    
    # --- Goods market metrics ---
    goods_availability = env_state.get('goods_availability', 1.0)
    consumer_price = env_state.get('consumer_goods_price', 1.0)
    capital_price = env_state.get('capital_goods_price', 2.0)
    energy_price = env_state.get('energy_price', 1.0)
    
    # --- Capital and Technology metrics ---
    avg_technology_level = env_state.get('avg_technology_level', 1.0)
    capital_goods_demand = env_state.get('capital_goods_demand', 0.0)
    capital_goods_supply = env_state.get('capital_goods_supply', 0.0)
    
    # --- Energy and Environmental metrics ---
    energy_demand = env_state.get('energy_demand', 0.0)
    energy_supply = env_state.get('energy_supply', 0.0)
    carbon_emissions = env_state.get('carbon_emissions', 0.0)
    renewable_fraction = env_state.get('avg_renewable_fraction', 0.2)
    
    # --- Financial metrics ---
    interest_rate = env_state.get('interest_rate', 0.05)
    debt_to_gdp = env_state.get('debt_to_gdp', 0.0)
    
    # --- Welfare metrics ---
    avg_utility = env_state.get('avg_utility', 0.0)
    income_per_capita = env_state.get('income_per_capita', 0.0)
    
    # --- Calculate inequality (Gini coefficient) ---
    household_states = agent_states.get('households', {})
    gini_coefficient = 0.0
    
    if household_states and 'income' in household_states:
        # Get income array and sort it
        incomes = jnp.sort(household_states['income'])
        n = incomes.shape[0]
        
        if n > 0:
            # Calculate Gini using the formula based on ordered incomes, with protection against division by zero
            index = jnp.arange(1, n + 1)
            income_sum = jnp.sum(incomes)
            
            # Only calculate if income_sum is positive
            gini_coefficient = jnp.where(
                income_sum > 1e-6,
                2 * jnp.sum(index * incomes) / (n * income_sum) - (n + 1) / n,
                0.0
            )
    
    # --- External impact metrics ---
    climate_impact = env_state.get('climate_impact', 1.0)
    pandemic_impact = env_state.get('pandemic_impact', 1.0)
    
    # --- Sector-specific metrics ---
    # Calculate sector contributions to GDP
    consumer_sector_gdp = env_state.get('consumer_goods_sold', 0.0) * consumer_price
    capital_sector_gdp = env_state.get('capital_goods_sold', 0.0) * capital_price
    energy_sector_gdp = env_state.get('energy_sold', 0.0) * energy_price
    govt_sector_gdp = env_state.get('govt_spending', 0.0)
    
    # Sector shares (as percentage of GDP)
    total_gdp = consumer_sector_gdp + capital_sector_gdp + energy_sector_gdp + govt_sector_gdp
    consumer_sector_share = jnp.where(total_gdp > 0.1, (consumer_sector_gdp / total_gdp) * 100, 0.0)
    capital_sector_share = jnp.where(total_gdp > 0.1, (capital_sector_gdp / total_gdp) * 100, 0.0)
    energy_sector_share = jnp.where(total_gdp > 0.1, (energy_sector_gdp / total_gdp) * 100, 0.0)
    govt_sector_share = jnp.where(total_gdp > 0.1, (govt_sector_gdp / total_gdp) * 100, 0.0)
    
    # --- Calculate overall economic health index ---
    # Weighted combination of key indicators
    economic_health = (
        0.25 * (1.0 - unemployment_rate) +  # Lower unemployment is better
        0.15 * jnp.clip(gdp_growth * 10, -1.0, 1.0) +  # GDP growth contributes positively
        0.15 * (1.0 - jnp.abs(inflation_rate - 0.02) * 10) +  # Closer to 2% inflation is better
        0.10 * (1.0 - jnp.minimum(debt_to_gdp, 1.0)) +  # Lower debt-to-GDP is better
        0.10 * goods_availability +  # Better goods availability is better
        0.10 * avg_utility / jnp.maximum(0.1, 2.0) +  # Higher utility is better
        0.05 * (1.0 - jnp.minimum(gini_coefficient, 1.0)) +  # Lower inequality is better
        0.05 * avg_technology_level / 2.0 +  # Higher technology level is better
        0.05 * renewable_fraction  # Higher renewable energy fraction is better
    )
    
    # Scale to 0-100 range
    economic_health_index = jnp.clip(economic_health * 100, 0, 100)
    
    # --- Environmental sustainability index ---
    # Track progress towards a sustainable economy
    sustainability_index = (
        0.4 * renewable_fraction +  # Higher renewable fraction is better
        0.3 * (1.0 - jnp.minimum(carbon_emissions / 100.0, 1.0)) +  # Lower emissions is better
        0.2 * jnp.clip((avg_technology_level - 1.0) * 0.5, 0.0, 1.0) +  # More advanced tech is better
        0.1 * (1.0 - jnp.maximum(0.0, (climate_impact - 1.0)))  # Lower climate impact is better
    ) * 100  # Scale to 0-100
    
    # Return all metrics, with safeguards against NaN values
    return {
        # Core economic indicators
        'gdp': jnp.nan_to_num(gdp, nan=0.1),  # Replace NaN with 0.1
        'gdp_growth': jnp.nan_to_num(gdp_growth * 100, nan=0.0),  # as percentage
        'inflation': jnp.nan_to_num(inflation_rate * 100, nan=0.0),  # as percentage
        'unemployment': jnp.nan_to_num(unemployment_rate * 100, nan=0.0),  # as percentage
        'wage_rate': jnp.nan_to_num(wage_rate, nan=1.0),
        'interest_rate': jnp.nan_to_num(interest_rate * 100, nan=0.0),  # as percentage
        
        # Market conditions
        'goods_availability': jnp.nan_to_num(goods_availability * 100, nan=100.0),  # as percentage
        'labor_market_tightness': jnp.nan_to_num(labor_market_tightness, nan=1.0),
        'consumer_price': jnp.nan_to_num(consumer_price, nan=1.0),
        'capital_price': jnp.nan_to_num(capital_price, nan=2.0),
        'energy_price': jnp.nan_to_num(energy_price, nan=1.0),
        
        # Welfare indicators
        'utility': jnp.nan_to_num(avg_utility, nan=0.0),
        'income_per_capita': jnp.nan_to_num(income_per_capita, nan=0.0),
        'inequality': jnp.nan_to_num(gini_coefficient, nan=0.0),
        
        # Sectoral indicators
        'consumer_sector_share': jnp.nan_to_num(consumer_sector_share, nan=0.0),
        'capital_sector_share': jnp.nan_to_num(capital_sector_share, nan=0.0),
        'energy_sector_share': jnp.nan_to_num(energy_sector_share, nan=0.0),
        'govt_sector_share': jnp.nan_to_num(govt_sector_share, nan=0.0),
        
        # Capital and technology indicators
        'technology_level': jnp.nan_to_num(avg_technology_level, nan=1.0),
        'capital_investment': jnp.nan_to_num(capital_goods_demand, nan=0.0),
        
        # Energy and environmental indicators
        'energy_demand': jnp.nan_to_num(energy_demand, nan=0.0),
        'energy_supply': jnp.nan_to_num(energy_supply, nan=0.0),
        'renewable_share': jnp.nan_to_num(renewable_fraction * 100, nan=20.0),  # as percentage
        'carbon_emissions': jnp.nan_to_num(carbon_emissions, nan=0.0),
        'sustainability_index': jnp.nan_to_num(sustainability_index, nan=50.0),
        
        # Financial indicators
        'debt_to_gdp': jnp.nan_to_num(debt_to_gdp * 100, nan=0.0),  # as percentage
        
        # External impacts
        'climate_impact': jnp.nan_to_num((1.0 - (climate_impact - 1.0)) * 100, nan=0.0),  # Convert to damage percentage
        'pandemic_impact': jnp.nan_to_num((1.0 - pandemic_impact) * 100, nan=0.0),  # Convert to damage percentage
        
        # Composite indicators
        'economic_health': jnp.nan_to_num(economic_health_index, nan=50.0)  # Default to middle value if NaN
    }

# ------------- MODEL CREATION -------------

def create_economy_model(
    num_households=1000,
    num_consumer_firms=50,
    num_capital_firms=20,
    num_energy_firms=10,
    enable_climate_module=False,
    enable_pandemic_module=False,
    tax_rate=0.2,
    interest_rate=0.05,
    energy_price=1.0,
    wage_rate=1.0,
    household_params=None,
    consumer_firm_params=None,
    capital_firm_params=None,
    energy_firm_params=None,
    govt_params=None,
    bank_params=None,
    seed=42,
    params=None,
    config=None
):
    """Create the economy model with specified parameters.
    
    This function sets up the complete economy model with all agent types,
    initial environmental state, and configuration parameters.
    
    Args:
        num_households: Number of household agents
        num_consumer_firms: Number of consumer goods firms
        num_capital_firms: Number of capital goods firms
        num_energy_firms: Number of energy firms
        enable_climate_module: Whether to activate climate impacts
        enable_pandemic_module: Whether to activate pandemic impacts
        tax_rate: Base tax rate
        interest_rate: Initial interest rate
        energy_price: Initial energy price
        wage_rate: Initial wage rate
        household_params: Parameters for household agents
        consumer_firm_params: Parameters for consumer firms
        capital_firm_params: Parameters for capital firms
        energy_firm_params: Parameters for energy firms
        govt_params: Parameters for government
        bank_params: Parameters for banks
        seed: Random seed
        params: Additional parameters passed through calibration tools
        config: Model configuration (if None, will create default)
        
    Returns:
        Configured Model instance ready for simulation
    """
    # Handle params from calibration if provided
    if params is not None:
        # Extract calibrated parameters if available
        tax_rate = params.get('tax_rate', tax_rate)
        interest_rate = params.get('interest_rate', interest_rate)
        energy_price = params.get('energy_price', energy_price)
        
        # These could override specific counts if needed
        num_households = params.get('num_households', num_households)
        num_consumer_firms = params.get('num_consumer_firms', num_consumer_firms)
        num_capital_firms = params.get('num_capital_firms', num_capital_firms)
        num_energy_firms = params.get('num_energy_firms', num_energy_firms)
    
    # Use provided config or create default
    if config is None:
        config = ModelConfig(
            seed=seed,
            steps=100,  # Default simulation length
            track_history=True,
            collect_interval=1  # Collect data every step
        )
    
    # Create model with environment update and metrics functions
    model = Model(
        params={
            'enable_climate_module': enable_climate_module,
            'enable_pandemic_module': enable_pandemic_module,
            'tax_rate': tax_rate,
            'interest_rate': interest_rate,
            'energy_price': energy_price,
            'wage_rate': wage_rate,
            'num_households': num_households,
            'num_consumer_firms': num_consumer_firms,
            'num_capital_firms': num_capital_firms,
            'num_energy_firms': num_energy_firms,
            # Pass through additional specified parameters
            **(params or {})
        },
        config=config,
        update_state_fn=update_environment,
        metrics_fn=compute_metrics
    )
    
    # --- Set up household agents ---
    household_defaults = {
        'initial_savings': 1000.0,
        'initial_income': 100.0,
        'propensity_to_consume': 0.8,
        'propensity_to_save': 0.1
    }
    
    # Override defaults with provided parameters
    if household_params:
        household_defaults.update(household_params)
    
    if params:  # Apply calibrated parameters if available
        for param_name in ['propensity_to_consume', 'propensity_to_save']:
            if param_name in params:
                household_defaults[param_name] = params[param_name]
    
    # Create household agent type and collection
    household_agent_type = Household(**household_defaults)
    
    households = AgentCollection(
        agent_type=household_agent_type,
        num_agents=num_households
    )
    
    # Add households to model
    model.add_agent_collection('households', households)
    
    # --- Set up consumer goods firms ---
    if num_consumer_firms > 0:
        consumer_firm_defaults = {
            'initial_capital': 1000.0,
            'initial_cash': 500.0,
            'production_efficiency': 1.0,
            'markup_rate': 0.2
        }
        
        # Override defaults with provided parameters
        if consumer_firm_params:
            consumer_firm_defaults.update(consumer_firm_params)
        
        if params:  # Apply calibrated parameters if available
            for param_name in ['production_efficiency', 'markup_rate']:
                if param_name in params:
                    consumer_firm_defaults[param_name] = params[param_name]
        
        # Create consumer goods firm agent type and collection
        consumer_firm_agent_type = ConsumerGoodsFirm(**consumer_firm_defaults)
        
        consumer_firms = AgentCollection(
            agent_type=consumer_firm_agent_type,
            num_agents=num_consumer_firms
        )
        
        # Add consumer firms to model
        model.add_agent_collection('consumer_firms', consumer_firms)
    
    # --- Set up capital goods firms ---
    if num_capital_firms > 0:
        capital_firm_defaults = {
            'initial_capital': 1500.0,
            'initial_cash': 800.0,
            'production_efficiency': 1.2,
            'r_and_d_intensity': 0.1,
            'markup_rate': 0.25
        }
        
        # Override defaults with provided parameters
        if capital_firm_params:
            capital_firm_defaults.update(capital_firm_params)
        
        if params:  # Apply calibrated parameters if available
            for param_name in ['production_efficiency', 'markup_rate', 'r_and_d_intensity']:
                if param_name in params:
                    capital_firm_defaults[param_name] = params[param_name]
        
        # Create capital goods firm agent type and collection
        capital_firm_agent_type = CapitalGoodsFirm(**capital_firm_defaults)
        
        capital_firms = AgentCollection(
            agent_type=capital_firm_agent_type,
            num_agents=num_capital_firms
        )
        
        # Add capital firms to model
        model.add_agent_collection('capital_firms', capital_firms)
    
    # --- Set up energy firms ---
    if num_energy_firms > 0:
        energy_firm_defaults = {
            'initial_capital': 2000.0,
            'initial_cash': 1000.0,
            'production_efficiency': 1.5,
            'renewable_fraction': 0.2,
            'markup_rate': 0.15
        }
        
        # Override defaults with provided parameters
        if energy_firm_params:
            energy_firm_defaults.update(energy_firm_params)
        
        if params:  # Apply calibrated parameters if available
            for param_name in ['production_efficiency', 'markup_rate', 'renewable_fraction']:
                if param_name in params:
                    energy_firm_defaults[param_name] = params[param_name]
        
        # Create energy firm agent type and collection
        energy_firm_agent_type = EnergyFirm(**energy_firm_defaults)
        
        energy_firms = AgentCollection(
            agent_type=energy_firm_agent_type,
            num_agents=num_energy_firms
        )
        
        # Add energy firms to model
        model.add_agent_collection('energy_firms', energy_firms)
    
    # --- Set up government ---
    # Government implementation would go here
    # For now we'll simulate its effects through environmental variables
    
    # --- Set up banking sector ---
    # Banking sector implementation would go here
    # For now we'll simulate its effects through environmental variables
    
    # --- Initialize environmental state with realistic starting values ---
    
    # Basic economic variables
    model.add_env_state('time_step', 0)
    model.add_env_state('wage_rate', wage_rate)
    model.add_env_state('price_level', 1.0)
    model.add_env_state('interest_rate', interest_rate)
    model.add_env_state('tax_rate', tax_rate)
    model.add_env_state('energy_price', energy_price)
    model.add_env_state('fossil_fuel_price', 0.8)  # Price of fossil fuels
    
    # Climate policy parameters
    model.add_env_state('climate_policy_strength', 0.2)  # Moderate climate policy
    model.add_env_state('carbon_price', 0.1)  # Carbon tax/price
    model.add_env_state('renewable_subsidy', 0.05)  # Subsidy for renewable energy
    
    # Initial economic activity - reasonable non-zero starting values
    initial_gdp = num_households * household_defaults['initial_income'] * 0.8
    model.add_env_state('gdp', initial_gdp)
    model.add_env_state('inflation_rate', 0.02)  # Start with 2% inflation
    
    # Labor market - realistic starting conditions
    model.add_env_state('job_market_condition', 1.0)
    model.add_env_state('employment_rate', 0.95)
    model.add_env_state('unemployment_rate', 0.05)
    model.add_env_state('total_labor_supply', num_households * 0.95)  # 95% employment
    model.add_env_state('total_labor_demand', num_households * 0.95)  # Matching demand
    
    # Goods markets - initial balance
    average_consumer_firm_production = 20.0  # Average production per firm
    consumer_total_production = num_consumer_firms * average_consumer_firm_production
    
    average_capital_firm_production = 10.0  # Capital goods firms produce fewer units
    capital_total_production = num_capital_firms * average_capital_firm_production
    
    average_energy_firm_production = 50.0  # Energy firms produce more units
    energy_total_production = num_energy_firms * average_energy_firm_production
    
    # Consumer goods market
    model.add_env_state('consumer_goods_price', 1.0)
    model.add_env_state('consumer_goods_supply', consumer_total_production)
    model.add_env_state('consumer_goods_demand', consumer_total_production * 0.9)  # Slight surplus
    model.add_env_state('consumer_goods_inventory', consumer_total_production * 0.1)
    
    # Capital goods market
    model.add_env_state('capital_goods_price', 2.0)
    model.add_env_state('capital_goods_supply', capital_total_production)
    model.add_env_state('capital_goods_demand', capital_total_production * 0.8)
    model.add_env_state('capital_goods_inventory', capital_total_production * 0.2)
    
    # Energy market
    model.add_env_state('energy_supply', energy_total_production)
    model.add_env_state('household_energy_demand', energy_total_production * 0.3)
    model.add_env_state('consumer_firms_energy_usage', energy_total_production * 0.4)
    model.add_env_state('capital_firms_energy_usage', energy_total_production * 0.3)
    model.add_env_state('goods_availability', 1.0)
    
    # Investment flows - initial investment demand
    model.add_env_state('consumer_firms_investment', capital_total_production * 0.5)
    model.add_env_state('energy_firms_investment', capital_total_production * 0.3)
    
    # Financial markets - healthy starting point
    total_savings = num_households * household_defaults['initial_savings']
    model.add_env_state('total_savings', total_savings)
    model.add_env_state('total_deposits', total_savings * 0.7)  # 70% of savings in banks
    model.add_env_state('total_loans', total_savings * 0.5)  # 50% of savings as loans
    
    # Fiscal variables - reasonable starting public finances
    model.add_env_state('tax_revenue', initial_gdp * tax_rate)
    model.add_env_state('govt_spending', initial_gdp * tax_rate * 1.1)  # Slight deficit
    model.add_env_state('public_debt', initial_gdp * 0.6)  # 60% debt-to-GDP ratio
    model.add_env_state('debt_to_gdp', 0.6)
    
    # Welfare metrics - decent starting point
    model.add_env_state('avg_utility', 1.0)
    model.add_env_state('income_per_capita', household_defaults['initial_income'])
    
    # External modules
    model.add_env_state('climate_impact', 1.0)
    model.add_env_state('pandemic_impact', 1.0)
    model.add_env_state('pandemic_infected_rate', 0.0)
    
    # Climate module variables
    model.add_env_state('climate_trend', 0.0)
    model.add_env_state('extreme_event_magnitude', 0.0)
    
    return model

# ------------- SIMULATION FUNCTIONS -------------

def run_simulation(args):
    """Run the main economic simulation.
    
    Sets up and runs the economic model with the specified parameters,
    collects results, and visualizes key economic indicators.
    
    Args:
        args: Command line arguments with simulation parameters
    
    Returns:
        Results dictionary with metrics from the simulation
    """
    print("Running advanced economic simulation...")
    
    # Create model with command line parameters
    model = create_economy_model(
        num_households=args.households,
        num_consumer_firms=args.consumer_firms,
        num_capital_firms=args.capital_firms,
        num_energy_firms=args.energy_firms,
        enable_climate_module=args.climate,
        enable_pandemic_module=args.pandemic,
        tax_rate=args.tax_rate,
        interest_rate=args.interest_rate,
        seed=args.seed
    )
    
    # Run simulation
    results = model.run()
    
    # Display key results
    if results:
        print("\nSimulation completed. Final economic indicators:")
        
        # Get final values
        final_idx = -1  # Last time step
        
        print(f"GDP: {results['gdp'][final_idx]:.4f}")
        print(f"GDP Growth: {results['gdp_growth'][final_idx]:.2f}%")
        print(f"Unemployment: {results['unemployment'][final_idx]:.2f}%")
        print(f"Inflation: {results['inflation'][final_idx]:.2f}%")
        print(f"Wage Rate: {results['wage_rate'][final_idx]:.4f}")
        print(f"Interest Rate: {results['interest_rate'][final_idx]:.2f}%")
        print(f"Income Inequality (Gini): {results['inequality'][final_idx]:.4f}")
        print(f"Economic Health Index: {results['economic_health'][final_idx]:.1f}/100")
        
        # Plot results
        plot_results(results)
    
    return results

def run_sensitivity(args):
    """Run sensitivity analysis on key parameters."""
    print("Running sensitivity analysis...")
    
    # Define the model factory function for sensitivity analysis
    def sensitivity_model_factory(params=None, config=None):
        """Create a model with the given parameters for sensitivity analysis."""
        return create_economy_model(
            num_households=min(args.households, 200),  # Use smaller populations for speed
            num_consumer_firms=min(args.consumer_firms, 10),
            num_capital_firms=min(args.capital_firms, 5),
            num_energy_firms=min(args.energy_firms, 3),
            enable_climate_module=args.climate,
            enable_pandemic_module=args.pandemic,
            seed=args.seed,
            params=params,
            config=config
        )
    
    # Create sensitivity analysis
    sensitivity = SensitivityAnalysis(
        model_factory=sensitivity_model_factory,
        param_ranges={
            'tax_rate': (0.1, 0.4),
            'interest_rate': (0.01, 0.1),
            'propensity_to_consume': (0.6, 0.9),
            'energy_price': (0.5, 2.0)
        },
        metrics_of_interest=[
            'gdp', 'unemployment', 'inflation', 'inequality', 'economic_health'
        ],
        num_samples=10,  # For demonstration - increase for real analysis
        seed=args.seed
    )
    
    # Run sensitivity analysis
    results = sensitivity.run()
    
    # Calculate Sobol indices
    indices = sensitivity.sobol_indices()
    
    # Display results
    print("\nSensitivity Analysis Results:")
    print("Parameter influence on key metrics (Sobol indices):")
    
    for metric in indices:
        print(f"\n{metric.upper()} is most influenced by:")
        sorted_params = sorted(
            [(param, indices[metric][param]['first_order']) 
             for param in indices[metric]],
            key=lambda x: x[1],
            reverse=True
        )
        
        for param, importance in sorted_params:
            print(f"  {param}: {importance:.4f}")
    
    # Visualization of sensitivity would be added here
    
    return results

def run_policy_experiment(args):
    """Test different policy interventions."""
    print("Running policy experiment...")
    
    # List of policy scenarios to test
    policy_scenarios = [
        {"name": "Baseline", "tax_rate": args.tax_rate, "interest_rate": args.interest_rate},
        {"name": "High Tax", "tax_rate": args.tax_rate + 0.1, "interest_rate": args.interest_rate},
        {"name": "Low Tax", "tax_rate": max(0.05, args.tax_rate - 0.1), "interest_rate": args.interest_rate},
        {"name": "High Interest", "tax_rate": args.tax_rate, "interest_rate": args.interest_rate + 0.02},
        {"name": "Low Interest", "tax_rate": args.tax_rate, "interest_rate": max(0.01, args.interest_rate - 0.02)},
    ]
    
    # Results for each policy
    policy_results = {}
    
    # Run simulations for each policy scenario
    for policy in policy_scenarios:
        print(f"\nRunning simulation for policy: {policy['name']}")
        
        # Create model with policy parameters
        model = create_economy_model(
            num_households=min(args.households, 200),  # Use smaller populations for speed
            num_consumer_firms=min(args.consumer_firms, 10),
            num_capital_firms=min(args.capital_firms, 5),
            num_energy_firms=min(args.energy_firms, 3),
            enable_climate_module=args.climate,
            enable_pandemic_module=args.pandemic,
            tax_rate=policy['tax_rate'],
            interest_rate=policy['interest_rate'],
            seed=args.seed
        )
        
        # Run simulation
        results = model.run()
        
        # Store results
        policy_results[policy['name']] = results
        
        # Display key final results
        if results:
            final_idx = -1
            print(f"  GDP Growth: {results['gdp_growth'][final_idx]:.2f}%")
            print(f"  Unemployment: {results['unemployment'][final_idx]:.2f}%")
            print(f"  Economic Health: {results['economic_health'][final_idx]:.1f}/100")
    
    # Compare policy outcomes
    print("\nPolicy Comparison:")
    metric = 'economic_health'  # Key metric for comparison
    
    for policy_name, results in policy_results.items():
        final_value = results[metric][-1]
        print(f"  {policy_name}: {final_value:.2f}")
    
    # Policy visualization would be added here
    
    return policy_results

def run_shock_experiment(args):
    """Test resilience to various shocks."""
    print("Running shock resilience experiment...")
    
    # Define shock scenarios
    shock_scenarios = [
        {"name": "No Shock", "pandemic_active": False, "pandemic_severity": 0.0},
        {"name": "Mild Pandemic", "pandemic_active": True, "pandemic_severity": 0.3},
        {"name": "Severe Pandemic", "pandemic_active": True, "pandemic_severity": 0.8},
    ]
    
    # Results for each shock
    shock_results = {}
    
    # Run simulations for each shock scenario
    for shock in shock_scenarios:
        print(f"\nRunning simulation for scenario: {shock['name']}")
        
        # Configure model with shock parameters
        params = {
            "pandemic_active": shock["pandemic_active"],
            "pandemic_severity": shock["pandemic_severity"],
            "enable_pandemic_module": shock["pandemic_active"]
        }
        
        # Create model
        model = create_economy_model(
            num_households=min(args.households, 200),  # Use smaller populations for speed
            num_consumer_firms=min(args.consumer_firms, 10),
            num_capital_firms=min(args.capital_firms, 5),
            num_energy_firms=min(args.energy_firms, 3),
            enable_pandemic_module=shock["pandemic_active"],
            seed=args.seed,
            params=params
        )
        
        # Run simulation
        results = model.run()
        
        # Store results
        shock_results[shock['name']] = results
        
        # Display key results focusing on resilience
        if results:
            final_idx = -1
            print(f"  GDP Growth: {results['gdp_growth'][final_idx]:.2f}%")
            print(f"  Unemployment: {results['unemployment'][final_idx]:.2f}%")
            print(f"  Pandemic Impact: {results['pandemic_impact'][final_idx]:.2f}%")
    
    # Shock resilience visualization would be added here
    
    return shock_results

# ------------- VISUALIZATION -------------

def plot_results(results, title="Economic Simulation Results"):
    """Plot key economic indicators from simulation results.
    
    Creates visualization of important economic metrics over time.
    
    Args:
        results: Results dictionary from model.run()
        title: Plot title
    """
    # Check if we have results to plot
    if not results or 'step' not in results:
        print("No results to plot")
        return
    
    # Create figure with subplots
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
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
    
    # Plot 3: Wages and Interest Rates
    ax3 = axs[1, 0]
    ax3.plot(time_steps, results['wage_rate'], 'c-', label='Wage Rate')
    ax3.set_ylabel('Wage Rate', color='c')
    ax3.tick_params(axis='y', labelcolor='c')
    
    ax3b = ax3.twinx()
    ax3b.plot(time_steps, results['interest_rate'], 'y-', label='Interest Rate %')
    ax3b.set_ylabel('Interest Rate %', color='y')
    ax3b.tick_params(axis='y', labelcolor='y')
    ax3.set_title('Factor Prices')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Inequality and Utility
    ax4 = axs[1, 1]
    ax4.plot(time_steps, results['inequality'], 'r-', label='Inequality (Gini)')
    ax4.set_ylabel('Inequality', color='r')
    ax4.tick_params(axis='y', labelcolor='r')
    
    ax4b = ax4.twinx()
    ax4b.plot(time_steps, results['utility'], 'b-', label='Avg Utility')
    ax4b.set_ylabel('Utility', color='b')
    ax4b.tick_params(axis='y', labelcolor='b')
    ax4.set_title('Welfare Indicators')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: External Impacts
    ax5 = axs[2, 0]
    climate_impact = results.get('climate_impact', [0] * len(time_steps))
    pandemic_impact = results.get('pandemic_impact', [0] * len(time_steps))
    
    ax5.plot(time_steps, climate_impact, 'g-', label='Climate Impact %')
    ax5.plot(time_steps, pandemic_impact, 'r-', label='Pandemic Impact %')
    ax5.set_ylabel('Impact %')
    ax5.set_title('External Shocks')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Overall Economic Health
    ax6 = axs[2, 1]
    ax6.plot(time_steps, results['economic_health'], 'k-', label='Economic Health')
    ax6.set_ylabel('Index (0-100)')
    ax6.set_title('Economic Health Index')
    ax6.grid(True, alpha=0.3)
    
    # Adjust layout and save/show
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    plt.savefig('economic_simulation_results.png', dpi=100, bbox_inches='tight')
    
    print("Plot saved as 'economic_simulation_results.png'")
    
    # Display plot
    plt.show()

# ------------- MAIN FUNCTION -------------

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Advanced economic model simulation")
    parser.add_argument("--simulation", action="store_true", help="Run main simulation")
    parser.add_argument("--sensitivity", action="store_true", help="Run sensitivity analysis")
    parser.add_argument("--policy", action="store_true", help="Run policy experiment")
    parser.add_argument("--shock", action="store_true", help="Run shock resilience experiment")
    parser.add_argument("--climate", action="store_true", help="Enable climate module")
    parser.add_argument("--pandemic", action="store_true", help="Enable pandemic module")
    parser.add_argument("--households", type=int, default=1000, help="Number of households")
    parser.add_argument("--consumer-firms", type=int, default=50, help="Number of consumer goods firms")
    parser.add_argument("--capital-firms", type=int, default=20, help="Number of capital goods firms")
    parser.add_argument("--energy-firms", type=int, default=10, help="Number of energy firms")
    parser.add_argument("--tax-rate", type=float, default=0.2, help="Tax rate")
    parser.add_argument("--interest-rate", type=float, default=0.05, help="Interest rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fast", action="store_true", help="Run with fewer agents for speed")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Adjust parameters for fast mode
    if args.fast:
        args.households = min(args.households, 200)
        args.consumer_firms = min(args.consumer_firms, 10)
        args.capital_firms = min(args.capital_firms, 5)
        args.energy_firms = min(args.energy_firms, 3)
    
    # Run appropriate simulation based on arguments
    if args.simulation or not (args.sensitivity or args.policy or args.shock):
        run_simulation(args)
    if args.sensitivity:
        run_sensitivity(args)
    if args.policy:
        run_policy_experiment(args)
    if args.shock:
        run_shock_experiment(args) 