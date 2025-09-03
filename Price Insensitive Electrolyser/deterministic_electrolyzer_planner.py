

"""
Deterministic Optimization Module for Electrolyzer Physical Planning.

This module creates a physically optimal production plan to meet a specific
final product (Compressed Hydrogen) target over a given time horizon. It does
not consider electricity prices but now manages the entire production chain,
including an electrolyzer, hydrogen storage, and a compressor.

Key features:
- Manages the full system to create a physically coherent operational plan.
- A `H2_Shortfall` variable allows the planner to miss the target if it is
  physically impossible, preventing the model from becoming infeasible.
- A very high penalty is applied to this shortfall, ensuring the model only
  misses the target as a last resort.
"""

import pyomo.environ as pyo
from pyomo.common.errors import ApplicationError
import pandas as pd
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

def log_gurobi_iis_deterministic(model, output_dir, filename_ilp="infeasible_planner_model.ilp"):
    """ Logs Gurobi IIS information if the deterministic model is infeasible. """
    # This logging function remains unchanged.
    iis_path = os.path.join(output_dir, filename_ilp)
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.splitext(filename_ilp)[0]
    lp_filename = os.path.join(output_dir, f"{base_filename}_debug.lp")
    print(f"INFO: Attempting to log Gurobi IIS for deterministic planner. Target IIS output file: {iis_path}")
    try:
        model.write(lp_filename, io_options={'symbolic_solver_labels': True})
        print(f"INFO: Saved LP file for debugging: {lp_filename}")
        print("\n" + "*"*20 + " GUROBI IIS INSTRUCTIONS (Deterministic Planner) " + "*"*20)
        print("INFO: To compute the Irreducible Inconsistent Subsystem (IIS)")
        print(f"INFO: and generate the ILP file ('{filename_ilp}'),")
        print("INFO: run the following command in your terminal where Gurobi is installed:")
        abs_iis_path = os.path.abspath(iis_path)
        abs_model_file_path = os.path.abspath(lp_filename)
        print(f"\ngurobi_cl ResultFile=\"{abs_iis_path}\" \"{abs_model_file_path}\"\n")
        print("*"* (40 + len(" GUROBI IIS INSTRUCTIONS (Deterministic Planner) ")))
        print("(Note: Ensure Gurobi command line tools are in your system's PATH.)\n")
    except Exception as write_e_lp:
        print(f"ERROR: Could not write LP file for IIS analysis: {write_e_lp}")


class DeterministicElectrolyzerPlanner:
    """
    Creates a deterministic physical plan to meet a hydrogen production target
    for the complex system.
    """
    def __init__(self, config=None):
        self.config = config or {}
        self.solver_config = self.config.get('solver', {'name': 'gurobi', 'threads': 0, 'options': {}})
        sim_config = self.config.get('simulation', {})
        self.output_dir = sim_config.get('output_dir', './electrolyzer_results_plan_first_complex')
        os.makedirs(self.output_dir, exist_ok=True)
        self.shortfall_penalty_factor = 1e9

    def _get_plant_abs_limit(self, plant_name, limit_key, default_if_nan=0.0):
        """ Safely retrieves an absolute limit for a plant from the definitions. """
        val = ABSOLUTE_PLANT_LIMITS.get(plant_name, {}).get(limit_key, default_if_nan)
        return default_if_nan if pd.isna(val) else val

    
    def plan_schedule_for_target(self, production_target_for_window, horizon_days,
                                 initial_plant_operating_levels,
                                 initial_storage_levels, # <-- NEW PARAMETER
                                 maintenance_schedule=None):
        """
        Builds and solves the deterministic planning model for the complex system.
        """
        opt_start_time = time.time()
        total_hours_in_window = 24 * horizon_days

        model = pyo.ConcreteModel(name="Electrolyzer_Deterministic_Planner_Complex_V2")

        
        model.H = pyo.Set(initialize=range(total_hours_in_window))
        model.H_plus_1 = pyo.Set(initialize=range(total_hours_in_window + 1))
        model.PLANTS_FLEX = pyo.Set(initialize=FLEXIBLE_PLANTS)
        model.MATERIALS = pyo.Set(initialize=MATERIALS_IN_STORAGE)

        model.PRODUCTION_TARGET = pyo.Param(initialize=production_target_for_window)

        # --- MODIFIED: Add variables for storage and on/off state ---
        model.P_plant = pyo.Var(model.PLANTS_FLEX, model.H, within=pyo.NonNegativeReals, initialize=0)
        model.Plant_Is_On_t = pyo.Var(model.PLANTS_FLEX, model.H, within=pyo.Binary, initialize=0)
        model.S_material = pyo.Var(model.MATERIALS, model.H_plus_1, within=pyo.NonNegativeReals, initialize=0)
        model.Energy_consumption_t = pyo.Var(model.H, within=pyo.NonNegativeReals, initialize=0)
        model.H2_produced_hourly = pyo.Var(model.H, within=pyo.NonNegativeReals, initialize=0)
        model.H2_Shortfall = pyo.Var(within=pyo.NonNegativeReals, initialize=0, doc="Amount of final product target missed (kg)")

 
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

        # Ramp constraints now apply to both plants
        @model.Constraint(model.PLANTS_FLEX, model.H)
        def ramp_up(model, p, t):
            ramp_limit = self._get_plant_abs_limit(p, 'Ramp_Up_Abs', 1e9)
            initial_op = initial_plant_operating_levels.get(p, 0.0)
            if t == 0:
                return model.P_plant[p, t] - initial_op <= ramp_limit
            return model.P_plant[p, t] - model.P_plant[p, t-1] <= ramp_limit

        @model.Constraint(model.PLANTS_FLEX, model.H)
        def ramp_down(model, p, t):
            ramp_limit = self._get_plant_abs_limit(p, 'Ramp_Down_Abs', 1e9)
            initial_op = initial_plant_operating_levels.get(p, 0.0)
            if t == 0:
                return initial_op - model.P_plant[p, t] <= ramp_limit
            return model.P_plant[p, t-1] - model.P_plant[p, t] <= ramp_limit


        @model.Constraint(model.MATERIALS)
        def storage_initial_level(model, m):
            return model.S_material[m, 0] == initial_storage_levels.get(m, 0.0)

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
        def hourly_energy_consumption_rule(model, t):
            demand = sum(model.P_plant[p, t] * abs(ELECTRICITY_INPUT_MAP.get(p, 0.0)) for p in model.PLANTS_FLEX)
            return model.Energy_consumption_t[t] == demand

        @model.Constraint(model.H)
        def hourly_h2_production_rule(model, t):

            h2_efficiency = PLANT_DEFINITIONS['Electrolyzer']['Outputs'].get('Hydrogen', 0)
            h2_prod = model.P_plant['Electrolyzer', t] * h2_efficiency
            return model.H2_produced_hourly[t] == h2_prod

        # --- MODIFIED: Target is now for the final product ---
        @model.Constraint(doc="Enforce the final product target for the window, allowing for a shortfall.")
        def enforce_total_production_target(model):

            total_final_product_produced = sum(model.P_plant['Compressor', t] for t in model.H)
            return total_final_product_produced + model.H2_Shortfall >= model.PRODUCTION_TARGET


        def objective_minimize_shortfall_penalty(model):
            """
            The objective is to minimize the penalty on the shortfall.
            This makes meeting the target the primary goal of the optimization.
            """
            return model.H2_Shortfall * self.shortfall_penalty_factor

        model.objective = pyo.Objective(rule=objective_minimize_shortfall_penalty, sense=pyo.minimize)

        # --- Solve ---
        solver_name = self.solver_config.get('name', 'gurobi')
        try:
            solver = pyo.SolverFactory(solver_name)
        except ApplicationError:
            print(f"ERROR: Solver '{solver_name}' not found.")
            return None, None

        solver_options = self.solver_config.get('options', {}).copy()
        for option, value in solver_options.items():
            try:
                solver.options[option] = value
            except:
                print(f"  Warning: Could not set planner solver option {option}={value}")

        results = None
        try:
            results = solver.solve(model, tee=False)
        except Exception as e:
            print(f"ERROR during deterministic planner solver execution: {e}")
            traceback.print_exc()
            return model, None

        if results and hasattr(results, 'solver'):
            term_cond = results.solver.termination_condition
            print(f"  Deterministic Planner Solver Status: {results.solver.status}, Termination: {term_cond}")
            if term_cond == pyo.TerminationCondition.infeasible:
                print("CRITICAL INFO: Deterministic Planner model reported as INFEASIBLE even with shortfall logic.")
                if solver_name.lower() == 'gurobi':
                    log_gurobi_iis_deterministic(model, self.output_dir, filename_ilp=f"planner_infeasible_h{horizon_days}.ilp")

        print(f"  Deterministic planning completed in {time.time() - opt_start_time:.2f}s.")
        return model, results
