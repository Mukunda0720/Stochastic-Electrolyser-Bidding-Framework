
"""
Deterministic Optimization Module for the Perfect Foresight Benchmark.

This module creates a single, globally optimal operational plan for the entire
simulation period, assuming perfect knowledge of future electricity prices.
Its purpose is to establish the maximum possible profit the system can achieve,
serving as an upper-bound benchmark for other, more realistic strategies.
"""

import pyomo.environ as pyo
from pyomo.common.errors import ApplicationError
import pandas as pd
import numpy as np
import os
import traceback
import time


from electrolyzer_definitions import (
    PLANT_DEFINITIONS,
    STORAGE_DEFINITIONS,
    MATERIALS_IN_STORAGE,
    FLEXIBLE_PLANTS,
    ABSOLUTE_PLANT_LIMITS,
    ELECTRICITY_INPUT_MAP
)

class PerfectInformationPlanner:
    """
    Creates a single, globally optimal schedule given perfect price information.
    """
    def __init__(self, config=None):
        self.config = config or {}
        self.solver_config = self.config.get('solver', {})
        sim_config = self.config.get('simulation', {})
        self.output_dir = sim_config.get('output_dir')
        os.makedirs(self.output_dir, exist_ok=True)

        # Load economic parameters from config
        self.hydrogen_price_config = self.config.get('hydrogen_price', {})
        self.p_hydrogen_input = self.hydrogen_price_config.get('value_eur_per_kg', 0.0)
        self.tou_tariff_config = self.config.get('time_of_use_tariff', {})
        self.tou_enabled = self.tou_tariff_config.get('enabled', False)
        self.tou_rate = self.tou_tariff_config.get('rate_eur_per_kw_month', 0.0)

        # Load production target
        self.electrolyzer_plant_config = self.config.get('electrolyzer_plant', {})
        self.target_h2_per_day = self.electrolyzer_plant_config.get('target_hydrogen_per_day', 0.0)

    def _get_plant_abs_limit(self, plant_name, limit_key, default_if_nan=0.0):
        """ Safely retrieves an absolute limit for a plant. """
        val = ABSOLUTE_PLANT_LIMITS.get(plant_name, {}).get(limit_key, default_if_nan)
        return default_if_nan if pd.isna(val) else val

    def _get_tou_weighting_factors(self, start_date, horizon_days):
        """Provides weighting factors for Time-of-Use tariff calculations."""
        # This helper function is identical to the one in the other managers
        weekday_weights = {1:[0.7,0.7,0.7,0.7,0.7,0.7,0.8,0.9,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.9,0.8],2:[0.7,0.7,0.7,0.7,0.7,0.7,0.8,0.9,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.9,0.8],3:[0.7,0.7,0.7,0.7,0.7,0.7,0.8,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,1.0,1.0,1.0,1.0,0.9,0.8,0.8],4:[0.7,0.7,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.7,0.6,0.6,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.8,0.8,0.8,0.8,0.8],5:[0.7,0.7,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.7,0.6,0.6,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.8,0.8,0.8,0.8,0.8],6:[0.7,0.7,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.7,0.6,0.6,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.8,0.8,0.8,0.8,0.8],7:[0.7,0.7,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.7,0.6,0.6,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.8,0.8,0.8,0.8,0.8],8:[0.7,0.7,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.7,0.6,0.6,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.8,0.8,0.8,0.8,0.8],9:[0.7,0.7,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.7,0.6,0.6,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.8,0.8,0.8,0.8,0.8],10:[0.7,0.7,0.7,0.7,0.7,0.7,0.8,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,1.0,1.0,1.0,1.0,0.9,0.8],11:[0.7,0.7,0.7,0.7,0.7,0.7,0.8,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,1.0,1.0,1.0,1.0,0.9,0.8],12:[0.7,0.7,0.7,0.7,0.7,0.7,0.8,0.9,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.9,0.8]}
        weekend_weights = [0.7,0.7,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.7,0.6,0.6,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.8,0.8,0.8,0.8,0.8]
        try:
            import holidays
            nl_holidays = holidays.country_holidays('NL', years=start_date.year)
        except ImportError: nl_holidays = {}
        weights = {}
        date_range = pd.date_range(start=start_date, periods=horizon_days * 24, freq='h')
        for t, timestamp in enumerate(date_range):
            is_weekend = timestamp.dayofweek >= 5; is_holiday = timestamp.date() in nl_holidays
            if is_weekend or is_holiday: weights[t] = weekend_weights[timestamp.hour]
            else: weights[t] = weekday_weights[timestamp.month][timestamp.hour]
        return weights

    def optimize_for_entire_period(self, actual_prices, start_date, total_days):
        """
        Builds and solves the perfect foresight optimization model.
        """
        opt_start_time = time.time()
        total_hours = total_days * 24

        model = pyo.ConcreteModel(name="Perfect_Information_Planner")

       
        model.H = pyo.Set(initialize=range(total_hours))
        model.H_plus_1 = pyo.Set(initialize=range(total_hours + 1))
        model.PLANTS_FLEX = pyo.Set(initialize=FLEXIBLE_PLANTS)
        model.MATERIALS = pyo.Set(initialize=MATERIALS_IN_STORAGE)

       
        model.Actual_Price = pyo.Param(model.H, initialize=actual_prices)
        model.P_hydrogen = pyo.Param(initialize=float(self.p_hydrogen_input))
        total_h2_target = self.target_h2_per_day * total_days
        model.TOTAL_H2_TARGET = pyo.Param(initialize=total_h2_target)

        if self.tou_enabled:
            hourly_tou_weights = self._get_tou_weighting_factors(start_date, total_days)
            model.TOU_Tariff_Rate = pyo.Param(initialize=self.tou_rate)
            model.TOU_Weight = pyo.Param(model.H, initialize=hourly_tou_weights)
            model.TotalDays = pyo.Param(initialize=float(total_days))

       
        model.P_plant = pyo.Var(model.PLANTS_FLEX, model.H, within=pyo.NonNegativeReals)
        model.Plant_Is_On_t = pyo.Var(model.PLANTS_FLEX, model.H, within=pyo.Binary)
        model.S_material = pyo.Var(model.MATERIALS, model.H_plus_1, within=pyo.NonNegativeReals)
        model.H2_produced_hourly = pyo.Var(model.H, within=pyo.NonNegativeReals)
        model.Total_Electricity_MWh = pyo.Var(model.H, within=pyo.NonNegativeReals)
        model.H2_Shortfall = pyo.Var(within=pyo.NonNegativeReals, initialize=0)

        if self.tou_enabled:
            model.E_peak_chargeable = pyo.Var(within=pyo.NonNegativeReals)



        @model.Constraint(model.PLANTS_FLEX, model.H)
        def plant_min_on_operation(model, p, t):
            min_op = self._get_plant_abs_limit(p, 'Min_Op_When_On_Abs', 0.0)
            return model.P_plant[p, t] >= min_op * model.Plant_Is_On_t[p, t]

        @model.Constraint(model.PLANTS_FLEX, model.H)
        def plant_min_absolute_operation(model, p, t):
            """Ensures plant operation is always above its absolute minimum floor."""
            min_op_abs = self._get_plant_abs_limit(p, 'Min_Op_Abs', 0.0)
            return model.P_plant[p, t] >= min_op_abs

        @model.Constraint(model.PLANTS_FLEX, model.H)
        def plant_max_operation(model, p, t):
            max_op = self._get_plant_abs_limit(p, 'Max_Op_Abs', 1e9)
            return model.P_plant[p, t] <= max_op * model.Plant_Is_On_t[p, t]

        @model.Constraint(model.PLANTS_FLEX, model.H)
        def ramp_up(model, p, t):
            ramp_limit = self._get_plant_abs_limit(p, 'Ramp_Up_Abs', 1e9)
            if t == 0: return pyo.Constraint.Skip # Initial state is handled by storage initial level
            return model.P_plant[p, t] - model.P_plant[p, t-1] <= ramp_limit

        @model.Constraint(model.PLANTS_FLEX, model.H)
        def ramp_down(model, p, t):
            ramp_limit = self._get_plant_abs_limit(p, 'Ramp_Down_Abs', 1e9)
            if t == 0: return pyo.Constraint.Skip
            return model.P_plant[p, t-1] - model.P_plant[p, t] <= ramp_limit

        @model.Constraint(model.MATERIALS)
        def storage_initial_level(model, m):
            return model.S_material[m, 0] == STORAGE_DEFINITIONS[m]['Initial_Level']

        @model.Constraint(model.MATERIALS, model.H)
        def storage_min_bound(model, m, t):
            return model.S_material[m, t+1] >= STORAGE_DEFINITIONS[m]['Min_Level']

        @model.Constraint(model.MATERIALS, model.H)
        def storage_max_bound(model, m, t):
            return model.S_material[m, t+1] <= STORAGE_DEFINITIONS[m]['Max_Level']

        @model.Constraint(model.MATERIALS, model.H)
        def storage_balance(model, m, t):
            production = model.H2_produced_hourly[t]
            consumption = model.P_plant['Compressor', t]
            return model.S_material[m, t+1] == model.S_material[m, t] + production - consumption

        @model.Constraint(model.H)
        def hourly_electricity_balance(model, t):
            demand = sum(model.P_plant[p, t] * abs(ELECTRICITY_INPUT_MAP.get(p, 0.0)) for p in model.PLANTS_FLEX)
            return model.Total_Electricity_MWh[t] == demand

        @model.Constraint(model.H)
        def hourly_h2_production_rule(model, t):
            h2_efficiency = PLANT_DEFINITIONS['Electrolyzer']['Outputs'].get('Hydrogen', 0)
            return model.H2_produced_hourly[t] == model.P_plant['Electrolyzer', t] * h2_efficiency

        @model.Constraint
        def enforce_total_production_target(model):
            total_final_product = sum(model.P_plant['Compressor', t] for t in model.H)
            return total_final_product + model.H2_Shortfall >= model.TOTAL_H2_TARGET

        if self.tou_enabled:
            @model.Constraint(model.H)
            def tou_peak_constraint(model, t):
                baseload_mw = self.electrolyzer_plant_config.get('baseload_mw', 0.0)
                total_site_consumption = model.Total_Electricity_MWh[t] + baseload_mw
                weighted_energy = total_site_consumption * model.TOU_Weight[t]
                return model.E_peak_chargeable >= weighted_energy

        
        def objective_rule_profit_maximization(model):
            market_energy_cost = sum(model.Total_Electricity_MWh[t] * model.Actual_Price[t] for t in model.H)
            
            tou_peak_charge = 0.0
            if self.tou_enabled:
                # Scale monthly rate to the simulation period
                scaling_factor = model.TotalDays / 30.4
                tou_peak_charge = (model.E_peak_chargeable * 1000 * model.TOU_Tariff_Rate) * scaling_factor

            total_cost = market_energy_cost + tou_peak_charge
            
            total_revenue = sum(model.P_plant['Compressor', t] for t in model.H) * model.P_hydrogen
            
           
            shortfall_penalty = model.H2_Shortfall * 1e9

            
            return total_cost - total_revenue + shortfall_penalty

        model.objective = pyo.Objective(rule=objective_rule_profit_maximization, sense=pyo.minimize)

        # --- Solve ---
        solver = pyo.SolverFactory(self.solver_config.get('name', 'gurobi'))
        solver.options.update(self.solver_config.get('options', {}))
        
        print("Solving the Perfect Information model for the entire period...")
        results = solver.solve(model, tee=True)
        print(f"Perfect Information model solved in {time.time() - opt_start_time:.2f}s.")

        return model, results
