# main stochastic optmisation module



import pyomo.environ as pyo
from pyomo.common.errors import ApplicationError
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import os
import traceback
import time


from electrolyzer_definitions import (
    PLANT_DEFINITIONS,
    STORAGE_DEFINITIONS as BASE_STORAGE_DEFINITIONS,
    MATERIALS_IN_STORAGE,
    FLEXIBLE_PLANTS,
    ABSOLUTE_PLANT_LIMITS,
    ELECTRICITY_INPUT_MAP
)
from utils_ramp import compute_fleet_symmetric_ramp_capability_mw

def log_gurobi_iis_eg(model, output_dir, filename_ilp="infeasible_model_eg.ilp"):
    """ Logs Gurobi IIS information if the model is infeasible. """
    # This function remains unchanged.
    iis_path = os.path.join(output_dir, filename_ilp)
    model_file_path = None
    model_format_used = None
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.splitext(filename_ilp)[0]
    lp_filename = os.path.join(output_dir, f"{base_filename}_debug.lp")
    mps_filename = os.path.join(output_dir, f"{base_filename}_debug.mps")

    print(f"INFO: Attempting to log Gurobi IIS. Target IIS output file: {iis_path}")
    try:
        model.write(lp_filename, io_options={'symbolic_solver_labels': True})
        model_file_path = lp_filename
        model_format_used = "LP"
        print(f"INFO: Saved LP file for debugging: {model_file_path}")
    except Exception:
        print(f"WARNING: Could not write LP file, trying MPS format...")
        try:
            model.write(mps_filename, io_options={'symbolic_solver_labels': True})
            model_file_path = mps_filename
            model_format_used = "MPS"
            print(f"INFO: Saved MPS file for debugging: {model_file_path}")
        except Exception as write_e_mps:
            print(f"ERROR: Could not write LP or MPS file for IIS analysis: {write_e_mps}")
            return

    if model_file_path and model_format_used:
        print("\n" + "*"*20 + " GUROBI IIS INSTRUCTIONS " + "*"*20)
        print("INFO: To compute the Irreducible Inconsistent Subsystem (IIS)")
        print(f"INFO: and generate the ILP file ('{filename_ilp}'),")
        print("INFO: run the following command in your terminal where Gurobi is installed:")
        abs_iis_path = os.path.abspath(iis_path)
        abs_model_file_path = os.path.abspath(model_file_path)
        print(f"\ngurobi_cl ResultFile=\"{abs_iis_path}\" \"{abs_model_file_path}\"\n")
        print("*"* (40 + len(" GUROBI IIS INSTRUCTIONS ")))
        print("(Note: Ensure Gurobi command line tools are in your system's PATH.)\n")
    else:
        print("INFO: Model file for IIS could not be saved. Cannot provide IIS command.")


class ElectrolyzerStochasticOptimizerEG:
    def __init__(self, config=None):
        self.config = config or {}
        self.solver_config = self.config.get('solver', {'name': 'gurobi', 'threads': 0, 'options': {}})
        self.scenario_data_cache = None
        sim_config = self.config.get('simulation', {})
        self.output_dir = sim_config.get('output_dir', './electrolyzer_results_v5_storage')
        os.makedirs(self.output_dir, exist_ok=True)

        self.eg_config = self.config.get('exclusive_group_bids', {})
        self.num_profiles_to_generate = self.eg_config.get('num_profiles_to_generate', 10)

        self.electrolyzer_plant_config = self.config.get('electrolyzer_plant', {})
        self.opportunity_cost_mwh_shortfall = self.electrolyzer_plant_config.get('opportunity_cost_mwh_shortfall', 158.0)

        self.initial_plant_operating_levels_param = None
        self.horizon_days_param = sim_config.get('planning_horizon', 2)
        self.implementation_period_days = int(sim_config.get('implementation_period', 1))

        self.objective_tuning_config = self.config.get('objective_tuning', {})
        self.enable_shortfall_penalty = self.objective_tuning_config.get('enable_production_shortfall_penalty', True)
        self.h2_opp_cost_per_kg = self.objective_tuning_config.get('h2_opportunity_cost_per_kg', 8.0)

        self.hydrogen_price_config = self.config.get('hydrogen_price', {})
        self.hydrogen_price_mode = self.hydrogen_price_config.get('mode', 'constant')
        self.p_hydrogen_input = self.hydrogen_price_config.get('value_eur_per_kg', 0.0)
        if self.p_hydrogen_input > 0:
            print(f"  OptimizerEG INFO (V5.1): Profit maximization ENABLED with P_hydrogen = {self.p_hydrogen_input} €/kg.")
        else:
            print("  OptimizerEG INFO (V5.1): Running in cost-minimization mode (P_hydrogen <= 0).")

        self.target_avg_h2_per_day_in_window_param = self.electrolyzer_plant_config.get('target_avg_h2_per_day_in_window', 0.0)
        self.redispatch_config = self.config.get('redispatch', {})
        self.redispatch_unused_energy_penalty_factor_param = float(self.redispatch_config.get('unused_energy_penalty_factor', 1e9))

        self.tou_tariff_config = self.config.get('time_of_use_tariff', {})
        self.tou_enabled = self.tou_tariff_config.get('enabled', False)
        self.tou_rate = self.tou_tariff_config.get('rate_eur_per_kw_month', 0.0)

    def _get_plant_abs_limit(self, plant_name, limit_key, default_if_nan=0.0):
        val = ABSOLUTE_PLANT_LIMITS.get(plant_name, {}).get(limit_key, default_if_nan)
        return default_if_nan if pd.isna(val) else val
    #touweights
    def _get_tou_weighting_factors(self, start_date, horizon_days):

        weekday_weights = {
            1:  [0.7,0.7,0.7,0.7,0.7,0.7, 0.7,0.8,0.9, 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0, 1.0,1.0,1.0, 1.0,1.0,0.9,0.8], 2:  [0.7,0.7,0.7,0.7,0.7,0.7, 0.7,0.8,0.9, 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0, 1.0,1.0,1.0, 1.0,1.0,0.9,0.8],
            3:  [0.7,0.7,0.7,0.7,0.7,0.7, 0.7,0.8,0.9, 0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9, 1.0,1.0,1.0, 1.0,0.9,0.8,0.8], 4:  [0.7,0.7,0.6,0.6,0.6,0.6, 0.6,0.7,0.8, 0.8,0.7,0.6,0.6,0.6,0.6,0.6,0.6, 0.7,0.8,0.8, 0.8,0.8,0.8,0.8],
            5:  [0.7,0.7,0.6,0.6,0.6,0.6, 0.6,0.7,0.8, 0.8,0.7,0.6,0.6,0.6,0.6,0.6,0.6, 0.7,0.8,0.8, 0.8,0.8,0.8,0.8], 6:  [0.7,0.7,0.6,0.6,0.6,0.6, 0.6,0.7,0.8, 0.8,0.7,0.6,0.6,0.6,0.6,0.6,0.6, 0.7,0.8,0.8, 0.8,0.8,0.8,0.8],
            7:  [0.7,0.7,0.6,0.6,0.6,0.6, 0.6,0.7,0.8, 0.8,0.7,0.6,0.6,0.6,0.6,0.6,0.6, 0.7,0.8,0.8, 0.8,0.8,0.8,0.8], 8:  [0.7,0.6,0.7,0.6,0.6,0.6, 0.6,0.7,0.8, 0.8,0.7,0.6,0.6,0.6,0.6,0.6,0.6, 0.7,0.8,0.8, 0.8,0.8,0.8,0.8],
            9:  [0.7,0.6,0.7,0.6,0.6,0.6, 0.6,0.7,0.8, 0.8,0.7,0.6,0.6,0.6,0.6,0.6,0.6, 0.7,0.8,0.8, 0.8,0.8,0.8,0.8], 10: [0.7,0.7,0.7,0.7,0.7,0.7, 0.7,0.8,0.9, 0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9, 1.0,1.0,1.0, 1.0,1.0,0.9,0.8],
            11: [0.7,0.7,0.7,0.7,0.7,0.7, 0.7,0.8,0.9, 0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9, 1.0,1.0,1.0, 1.0,1.0,0.9,0.8], 12: [0.7,0.7,0.7,0.7,0.7,0.7, 0.7,0.8,0.9, 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0, 1.0,1.0,1.0, 1.0,1.0,0.9,0.8]
        }
        weekend_weights = [0.7,0.6,0.7,0.6,0.6,0.6, 0.6,0.7,0.8, 0.8,0.7,0.6,0.6,0.6,0.6,0.6,0.6, 0.7,0.8,0.8, 0.8,0.8,0.8,0.8]
        try:
            import holidays
            nl_holidays = holidays.country_holidays('NL', years=start_date.year)
        except ImportError:
            print("Warning: 'holidays' library not found for TOU weights. Holiday weighting will be inaccurate.")
            nl_holidays = {}
        weights = {}
        first_month = start_date.month
        date_range = pd.date_range(start=start_date, periods=horizon_days * 24, freq='H')
        for t, timestamp in enumerate(date_range):
            is_weekend = timestamp.dayofweek >= 5
            is_holiday = timestamp.date() in nl_holidays
            if is_weekend or is_holiday:
                weights[t] = weekend_weights[timestamp.hour]
            else:
                weights[t] = weekday_weights[first_month][timestamp.hour]
        return weights

    def optimize_schedule(self, scenario_prices_df, horizon_days,
                          start_date_for_window,
                          current_initial_storage_levels=None,
                          initial_plant_operating_levels=None):
        
        self.horizon_days_param = horizon_days
        self.initial_plant_operating_levels_param = initial_plant_operating_levels or {p: self._get_plant_abs_limit(p, 'Min_Op_Abs', 0.0) for p in FLEXIBLE_PLANTS}

        risk_params = self.config.get('risk_management', {})
        cvar_alpha = risk_params.get('cvar_alpha', 0.95)
        cvar_weight = risk_params.get('cvar_weight', 0.0)
        total_hours_in_window = 24 * horizon_days
        H_window = range(total_hours_in_window)

        hourly_tou_weights = {}
        if self.tou_enabled:
            print("  OptimizerEG INFO: Time-of-Use (TOU) peak charge incentive is ENABLED.")
            try:
                hourly_tou_weights = self._get_tou_weighting_factors(start_date_for_window, horizon_days)
                if not hourly_tou_weights:
                    print("  OptimizerEG WARNING: TOU weighting factors helper returned empty. Disabling for this run.")
                    self.tou_enabled = False
            except Exception as e:
                 print(f"  OptimizerEG WARNING: Could not retrieve TOU weights due to: {e}. Disabling for this run.")
                 self.tou_enabled = False

        try:
            num_scenarios = len(scenario_prices_df)
            scenario_probs = scenario_prices_df["Scenario_Probability"].values
            if not np.isclose(np.sum(scenario_probs), 1.0): scenario_probs /= np.sum(scenario_probs)

            price_cols = [f"Hour_{h+1:02d}" for h in H_window]
            actual_scenario_market_prices = scenario_prices_df[price_cols].values
            average_internal_scenario_price_overall = np.average(actual_scenario_market_prices, weights=np.tile(scenario_probs, (total_hours_in_window,1)).T)
        except Exception as e:
            print(f"Error processing scenario data for EG optimizer: {e}"); traceback.print_exc(); return None, None, None

        model = pyo.ConcreteModel(name=f"Electrolyzer_EG_V5_Storage")
        model.S = pyo.Set(initialize=range(num_scenarios))
        model.H = pyo.Set(initialize=H_window)
        model.PLANTS_FLEX = pyo.Set(initialize=FLEXIBLE_PLANTS)
        model.PROFILES = pyo.Set(initialize=range(self.num_profiles_to_generate))
        
        model.H_plus_1 = pyo.Set(initialize=range(total_hours_in_window + 1))
        model.MATERIALS = pyo.Set(initialize=MATERIALS_IN_STORAGE)

        model.P_hydrogen = pyo.Param(initialize=float(self.p_hydrogen_input), mutable=False)
        self.TOTAL_H2_TARGET_FOR_WINDOW_CALCULATED = self.target_avg_h2_per_day_in_window_param * self.horizon_days_param
        model.TOTAL_H2_TARGET_FOR_WINDOW_PARAM = pyo.Param(initialize=self.TOTAL_H2_TARGET_FOR_WINDOW_CALCULATED)

        if self.tou_enabled:
            model.TOU_Tariff_Rate = pyo.Param(initialize=self.tou_rate)
            model.TOU_Weight = pyo.Param(model.H, initialize=hourly_tou_weights)
            model.HorizonDays = pyo.Param(initialize=float(horizon_days))

        model.P_profile_shape = pyo.Var(model.PROFILES, model.H, within=pyo.NonNegativeReals, initialize=0)
        model.U_profile_selected_s = pyo.Var(model.PROFILES, model.S, within=pyo.Binary, initialize=0)
        model.Actual_Accepted_E_s_t = pyo.Var(model.S, model.H, within=pyo.NonNegativeReals, initialize=0)
        model.P_plant_s = pyo.Var(model.S, model.PLANTS_FLEX, model.H, within=pyo.NonNegativeReals, initialize=0)
        model.H2_produced_hourly_in_window_s_t = pyo.Var(model.S, model.H, within=pyo.NonNegativeReals, initialize=0)
        model.Cost_s = pyo.Var(model.S, within=pyo.Reals)
        model.VaR = pyo.Var(within=pyo.Reals)
        model.Excess_Cost_s = pyo.Var(model.S, within=pyo.NonNegativeReals)
        
        model.Plant_Is_On_s_t = pyo.Var(model.S, model.PLANTS_FLEX, model.H, within=pyo.Binary, initialize=0)
        model.S_material_s = pyo.Var(model.S, model.MATERIALS, model.H_plus_1, within=pyo.NonNegativeReals, initialize=0)

        if self.enable_shortfall_penalty:
            model.H2_Shortfall_s = pyo.Var(model.S, within=pyo.NonNegativeReals, initialize=0)
        if self.tou_enabled:
            model.E_peak_chargeable_s = pyo.Var(model.S, within=pyo.NonNegativeReals, initialize=0)

        @model.Expression(model.S)
        def Revenue_s(model, s):
            price = model.P_hydrogen
            return price * sum(model.P_plant_s[s, 'Compressor', t] for t in model.H)

        @model.Expression(model.S)
        def NetCost_s(model, s):
            return model.Cost_s[s] - model.Revenue_s[s]

        @model.Constraint(model.S)
        def exclusive_profile_selection_rule(model, s):
            return sum(model.U_profile_selected_s[prof, s] for prof in model.PROFILES) <= 1

        M_energy_linking = max(1000.0, sum(self._get_plant_abs_limit(p, 'Max_Op_Abs', 0.0) for p in FLEXIBLE_PLANTS) * 1.5)
        @model.Constraint(model.S, model.H, model.PROFILES)
        def link_actual_energy_lb(model, s, t, prof):
            return model.Actual_Accepted_E_s_t[s,t] >= model.P_profile_shape[prof,t] - M_energy_linking * (1 - model.U_profile_selected_s[prof,s])
        @model.Constraint(model.S, model.H, model.PROFILES)
        def link_actual_energy_ub(model, s, t, prof):
            return model.Actual_Accepted_E_s_t[s,t] <= model.P_profile_shape[prof,t] + M_energy_linking * (1 - model.U_profile_selected_s[prof,s])
        @model.Constraint(model.S, model.H)
        def ensure_actual_energy_if_any_profile_selected(model, s, t):
            return model.Actual_Accepted_E_s_t[s,t] <= M_energy_linking * sum(model.U_profile_selected_s[prof,s] for prof in model.PROFILES)

        @model.Constraint(model.S, model.H)
        def plant_electricity_balance_s_t(model, s, t):
            demand = sum(
                model.P_plant_s[s, p_name, t] * abs(ELECTRICITY_INPUT_MAP.get(p_name, 0.0))
                for p_name in model.PLANTS_FLEX
            )
            return demand == model.Actual_Accepted_E_s_t[s,t]

        if self.tou_enabled:
            @model.Constraint(model.S, model.H)
            def tou_peak_constraint(model, s, t):
                weighted_energy = model.Actual_Accepted_E_s_t[s,t] * model.TOU_Weight[t]
                return model.E_peak_chargeable_s[s] >= weighted_energy
            
        @model.Constraint(model.S, model.PLANTS_FLEX, model.H)
        def plant_min_on_operation_rule(model, s, p, t):
            min_op_when_on_abs = self._get_plant_abs_limit(p, 'Min_Op_When_On_Abs', 0.0)
            return model.P_plant_s[s, p, t] >= min_op_when_on_abs * model.Plant_Is_On_s_t[s, p, t]
        
        @model.Constraint(model.S, model.PLANTS_FLEX, model.H)
        def plant_max_operation_rule(model, s, p, t):
            max_op_abs = self._get_plant_abs_limit(p, 'Max_Op_Abs', 1e9)
            return model.P_plant_s[s, p, t] <= max_op_abs * model.Plant_Is_On_s_t[s, p, t]

        @model.Constraint(model.S, model.PLANTS_FLEX, model.H)
        def plant_min_absolute_operation(model, s, p, t):
            min_op_abs = self._get_plant_abs_limit(p, 'Min_Op_Abs', 0.0)
            is_active = sum(model.U_profile_selected_s[prof,s] for prof in model.PROFILES)
            return model.P_plant_s[s, p, t] >= min_op_abs * is_active

        @model.Constraint(model.S, model.PLANTS_FLEX, model.H)
        def ramp_up_s_p_t(model, s, p, t):
            ramp_limit = self._get_plant_abs_limit(p, 'Ramp_Up_Abs', 1e9)
            if t == 0:
                initial_op = self.initial_plant_operating_levels_param.get(p, 0.0)
                return model.P_plant_s[s, p, t] - initial_op <= ramp_limit
            return model.P_plant_s[s, p, t] - model.P_plant_s[s, p, t-1] <= ramp_limit

        @model.Constraint(model.S, model.PLANTS_FLEX, model.H)
        def ramp_down_s_p_t(model, s, p, t):
            ramp_limit = self._get_plant_abs_limit(p, 'Ramp_Down_Abs', 1e9)
            if t == 0:
                initial_op = self.initial_plant_operating_levels_param.get(p, 0.0)
                return initial_op - model.P_plant_s[s, p, t] <= ramp_limit
            return model.P_plant_s[s, p, t-1] - model.P_plant_s[s, p, t] <= ramp_limit

        @model.Constraint(model.S, model.MATERIALS)
        def storage_initial_level_s_m(model, s, m):
            init_lvl = current_initial_storage_levels.get(m, BASE_STORAGE_DEFINITIONS[m]['Initial_Level'])
            return model.S_material_s[s, m, 0] == init_lvl

        @model.Constraint(model.S, model.MATERIALS, model.H)
        def storage_min_bound_s_m_t(model, s, m, t):
            return model.S_material_s[s, m, t+1] >= BASE_STORAGE_DEFINITIONS[m]['Min_Level']

        @model.Constraint(model.S, model.MATERIALS, model.H)
        def storage_max_bound_s_m_t(model, s, m, t):
            return model.S_material_s[s, m, t+1] <= BASE_STORAGE_DEFINITIONS[m]['Max_Level']

        @model.Constraint(model.S, model.MATERIALS, model.H)
        def storage_balance_s_m_t(model, s, m, t):
            production_in_hour = model.H2_produced_hourly_in_window_s_t[s,t]
            consumption_in_hour = model.P_plant_s[s, 'Compressor', t]
            return model.S_material_s[s, m, t+1] == model.S_material_s[s, m, t] + production_in_hour - consumption_in_hour

        @model.Constraint(model.S, model.H)
        def hourly_h2_production_rule_s_t(model, s, t):
            h2_prod = model.P_plant_s[s, 'Electrolyzer', t] * PLANT_DEFINITIONS['Electrolyzer']['Outputs'].get('Hydrogen', 0)
            return model.H2_produced_hourly_in_window_s_t[s,t] == h2_prod

        @model.Constraint(model.S)
        def constrain_total_h2_production_s(model, s):
            total_final_product = sum(model.P_plant_s[s, 'Compressor', t] for t in model.H)
            target = model.TOTAL_H2_TARGET_FOR_WINDOW_PARAM * sum(model.U_profile_selected_s[prof,s] for prof in model.PROFILES)
            if self.enable_shortfall_penalty:
                if self.TOTAL_H2_TARGET_FOR_WINDOW_CALCULATED > 1e-6:
                     return target - total_final_product <= model.H2_Shortfall_s[s]
                else:
                     return model.H2_Shortfall_s[s] == 0
            else:
                return total_final_product >= target

        @model.Constraint(model.S)
        def cost_rule_s(model, s):
            market_energy_cost = sum(model.Actual_Accepted_E_s_t[s,t] * actual_scenario_market_prices[s,t] for t in model.H)
            tou_peak_charge_incentive = 0.0
            if self.tou_enabled:
                scaling_factor = model.HorizonDays / 30.4
                tou_peak_charge_incentive = (model.E_peak_chargeable_s[s] * 1000 * model.TOU_Tariff_Rate) * scaling_factor
            return model.Cost_s[s] == market_energy_cost + tou_peak_charge_incentive

        @model.Constraint(model.S)
        def excess_cost_definition_s(model, s):
            return model.Excess_Cost_s[s] >= model.NetCost_s[s] - model.VaR

        def objective_rule_profit_cvar_eg(model):
            expected_net_cost = sum(scenario_probs[s] * model.NetCost_s[s] for s in model.S)
            cvar_component = 0.0
            if cvar_weight > 1e-6 and (1.0 - cvar_alpha) > 1e-9:
                cvar_component = cvar_weight * (model.VaR + (1.0 / (1.0 - cvar_alpha)) * sum(scenario_probs[s] * model.Excess_Cost_s[s] for s in model.S))
            shortfall_penalty_cost = 0.0
            if self.enable_shortfall_penalty and hasattr(model, 'H2_Shortfall_s'):
                shortfall_penalty_cost = sum(scenario_probs[s] * model.H2_Shortfall_s[s] * self.h2_opp_cost_per_kg for s in model.S)
            return expected_net_cost + cvar_component + shortfall_penalty_cost

        model.objective = pyo.Objective(rule=objective_rule_profit_cvar_eg, sense=pyo.minimize)

        solver = pyo.SolverFactory(self.solver_config.get('name', 'gurobi'))
        solver.options.update(self.solver_config.get('options', {}))
        results = solver.solve(model, tee=False)

        if results and hasattr(results, 'solver'):
            term_cond = results.solver.termination_condition
            if term_cond in [pyo.TerminationCondition.infeasible, pyo.TerminationCondition.infeasibleOrUnbounded]:
                print(f"CRITICAL (Main Opt): Solver reported status: {term_cond}")
                if self.solver_config.get('name', 'gurobi').lower() == 'gurobi':
                    log_gurobi_iis_eg(model, self.output_dir, filename_ilp=f"main_opt_infeasible_{start_date_for_window.strftime('%Y%m%d')}.ilp")

        self.scenario_data_cache = {
            'average_internal_scenario_price': average_internal_scenario_price_overall,
            'scenario_probs': scenario_probs, 'cvar_alpha': cvar_alpha, 'cvar_weight': cvar_weight
        }
        return model, results, self.scenario_data_cache

    def generate_eg_bids(self, model, time_index_for_window):
        # Profile pricing mechanisnm
        cfg = self.config.get('exclusive_group_bids', {})
        opp_cost_mwh = self.electrolyzer_plant_config.get('opportunity_cost_mwh_shortfall')
        avg_price = self.scenario_data_cache.get('average_internal_scenario_price', opp_cost_mwh)
        profiles_data = []
        for p_idx in model.PROFILES:
            hourly_mw = [round(pyo.value(model.P_profile_shape[p_idx, t]), 3) for t in model.H]
            selection_count = sum(pyo.value(model.U_profile_selected_s[p_idx, s]) for s in model.S)
            profiles_data.append({"profile_id": p_idx, "hourly_mw_values": hourly_mw, "total_energy_mwh": sum(hourly_mw), "selection_count": selection_count})
        ranked_profiles = sorted(profiles_data, key=lambda x: (x['selection_count'], x['total_energy_mwh']), reverse=True)
        t1_count = cfg.get('tier1_profile_count', 2); t1_factor = cfg.get('tier1_price_factor', 1.1)
        t2_count = cfg.get('tier2_profile_count', 4); t2_factor = cfg.get('tier2_price_factor', 0.9)
        t3_factor = cfg.get('tier3_price_factor', 0.7)
        threshold = opp_cost_mwh * (1.0 - (cfg.get('opportunity_cost_proximity_threshold_percent', 15.0) / 100.0))
        allow_premium = avg_price >= threshold
        for rank, profile in enumerate(ranked_profiles):
            if rank < t1_count: factor = t1_factor if allow_premium else min(1.0, t1_factor)
            elif rank < t1_count + t2_count: factor = t2_factor
            else: factor = t3_factor
            price = round(max(cfg.get('absolute_min_bid_price', 0.0), min(opp_cost_mwh * factor, cfg.get('absolute_max_bid_price', 4000.0))), 2)
            profile['profile_bid_price'] = price
            profile['rank_for_pricing'] = rank
        return sorted(ranked_profiles, key=lambda x: x['profile_id'])

    def optimize_energy_constrained_schedule_eg(self, start_datetime_obj, num_hours_to_redispatch,
                                             initial_storage_levels_for_redispatch,
                                             initial_plant_operating_levels_for_redispatch,
                                             hourly_cleared_energy_profile):
        H_redispatch = range(num_hours_to_redispatch)
        model = pyo.ConcreteModel(name=f"ElectrolyzerRedispatch_EG_V5_Storage")
        model.H = pyo.Set(initialize=H_redispatch)
        model.PLANTS_FLEX = pyo.Set(initialize=FLEXIBLE_PLANTS)
        model.H_plus_1 = pyo.Set(initialize=range(num_hours_to_redispatch + 1))
        model.MATERIALS = pyo.Set(initialize=MATERIALS_IN_STORAGE)

        model.P_plant = pyo.Var(model.PLANTS_FLEX, model.H, within=pyo.NonNegativeReals)
        model.H2_produced_hourly = pyo.Var(model.H, within=pyo.NonNegativeReals)
        model.Unused_Cleared_Energy_t = pyo.Var(model.H, within=pyo.NonNegativeReals)
        model.Min_Op_Violation_p_t = pyo.Var(model.PLANTS_FLEX, model.H, within=pyo.NonNegativeReals, initialize=0)
        model.Plant_Is_On_t = pyo.Var(model.PLANTS_FLEX, model.H, within=pyo.Binary)
        model.S_material = pyo.Var(model.MATERIALS, model.H_plus_1, within=pyo.NonNegativeReals)

        @model.Constraint(model.PLANTS_FLEX, model.H)
        def plant_min_on_operation_rd(model, p, t):
            min_op_when_on_abs = self._get_plant_abs_limit(p, 'Min_Op_When_On_Abs', 0.0)
            return model.P_plant[p, t] >= min_op_when_on_abs * model.Plant_Is_On_t[p, t]
        
        @model.Constraint(model.PLANTS_FLEX, model.H)
        def plant_operational_min_rd(model, p, t):
            min_op_abs = self._get_plant_abs_limit(p, 'Min_Op_Abs', 0.0)
            return model.P_plant[p, t] >= min_op_abs * model.Plant_Is_On_t[p, t]

        @model.Constraint(model.PLANTS_FLEX, model.H)
        def plant_max_operation_rd(model, p, t):
            max_op_abs = self._get_plant_abs_limit(p, 'Max_Op_Abs', 1e9)
            return model.P_plant[p, t] <= max_op_abs * model.Plant_Is_On_t[p, t]

        @model.Constraint(model.PLANTS_FLEX, model.H)
        def min_absolute_operation_rd(model, p, t):
            min_op = self._get_plant_abs_limit(p, 'Min_Op_Abs', 0.0)
            return model.P_plant[p, t] + model.Min_Op_Violation_p_t[p, t] >= min_op

        @model.Constraint(model.PLANTS_FLEX, model.H)
        def ramp_up_rd(model, p, t):
            ramp_limit = self._get_plant_abs_limit(p, 'Ramp_Up_Abs', 1e9)
            if t == 0: initial_op = initial_plant_operating_levels_for_redispatch.get(p, 0.0)
            else: initial_op = model.P_plant[p, t-1]
            return model.P_plant[p, t] - initial_op <= ramp_limit

        @model.Constraint(model.PLANTS_FLEX, model.H)
        def ramp_down_rd(model, p, t):
            ramp_limit = self._get_plant_abs_limit(p, 'Ramp_Down_Abs', 1e9)
            if t == 0: initial_op = initial_plant_operating_levels_for_redispatch.get(p, 0.0)
            else: initial_op = model.P_plant[p, t-1]
            return initial_op - model.P_plant[p, t] <= ramp_limit

        @model.Constraint(model.MATERIALS)
        def storage_initial_level_rd_m(model, m):
            return model.S_material[m, 0] == initial_storage_levels_for_redispatch[m]
        @model.Constraint(model.MATERIALS, model.H)
        def storage_min_bound_rd_m_t(model, m, t):
            return model.S_material[m, t+1] >= BASE_STORAGE_DEFINITIONS[m]['Min_Level']
        @model.Constraint(model.MATERIALS, model.H)
        def storage_max_bound_rd_m_t(model, m, t):
            return model.S_material[m, t+1] <= BASE_STORAGE_DEFINITIONS[m]['Max_Level']
        @model.Constraint(model.MATERIALS, model.H)
        def storage_balance_rd_m_t(model, m, t):
            production = model.H2_produced_hourly[t]
            consumption = model.P_plant['Compressor', t]
            return model.S_material[m,t+1] == model.S_material[m,t] + production - consumption

        @model.Constraint(model.H)
        def fixed_hourly_electricity_rd_t(model, t):
            demand = sum(model.P_plant[p, t] * abs(ELECTRICITY_INPUT_MAP.get(p, 0.0)) for p in model.PLANTS_FLEX)
            return demand == hourly_cleared_energy_profile[t] - model.Unused_Cleared_Energy_t[t]

        @model.Constraint(model.H)
        def hourly_h2_production_rule_rd_t(model, t):
            h2_prod = model.P_plant['Electrolyzer',t]*PLANT_DEFINITIONS['Electrolyzer']['Outputs'].get('Hydrogen',0)
            return model.H2_produced_hourly[t] == h2_prod

       
        def objective_rd(model):
            # Calculate total revenue from selling the final product (Compressed_Hydrogen)
            total_revenue_from_h2 = sum(model.P_plant['Compressor', t] for t in model.H) * self.p_hydrogen_input
            
            # Penalties remain the same and high for unused energy
            unused_energy_penalty = sum(model.Unused_Cleared_Energy_t[t] * self.redispatch_unused_energy_penalty_factor_param for t in model.H)
            violation_penalty_cost = 1e7
            total_min_op_violation_penalty = sum(model.Min_Op_Violation_p_t[p, t] * violation_penalty_cost for p in model.PLANTS_FLEX for t in model.H)
            

            return total_revenue_from_h2 - unused_energy_penalty - total_min_op_violation_penalty


        model.objective = pyo.Objective(rule=objective_rd, sense=pyo.maximize)

        solver = pyo.SolverFactory(self.solver_config.get('name', 'gurobi'))
        solver.options.update(self.solver_config.get('options', {}))
        results = solver.solve(model, tee=False)

        if results and hasattr(results, 'solver'):
            term_cond = results.solver.termination_condition
            if term_cond in [pyo.TerminationCondition.infeasible, pyo.TerminationCondition.infeasibleOrUnbounded]:
                print(f"CRITICAL (Re-dispatch): Solver reported status: {term_cond}")
                if self.solver_config.get('name', 'gurobi').lower() == 'gurobi':
                    log_gurobi_iis_eg(model, self.output_dir, filename_ilp=f"redispatch_infeasible_{start_datetime_obj.strftime('%Y%m%d')}.ilp")
            if term_cond in [pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible, pyo.TerminationCondition.locallyOptimal]:
                total_violation = sum(pyo.value(model.Min_Op_Violation_p_t[p,t]) for p in model.PLANTS_FLEX for t in model.H)
                if total_violation > 1e-4:
                    print(f"  Re-dispatch WARNING: Minimum operation constraint was violated by a total of {total_violation:.4f} to maintain feasibility.")
        return model, results


def process_stochastic_main_schedule_eg(hourly_prices_path, start_date, horizon_days,
                                        output_dir, config, solver_config=None,
                                        initial_storage_levels=None,
                                        initial_plant_operating_levels=None):
    
    try:
        scenario_df = pd.read_csv(hourly_prices_path)
        optimizer_eg = ElectrolyzerStochasticOptimizerEG(config)
        optimizer_eg.solver_config = solver_config or optimizer_eg.solver_config
        optimizer_eg.output_dir = output_dir

        model, results, scenario_data = optimizer_eg.optimize_schedule(
            scenario_prices_df=scenario_df, horizon_days=horizon_days,
            start_date_for_window=start_date,
            current_initial_storage_levels=initial_storage_levels,
            initial_plant_operating_levels=initial_plant_operating_levels,
        )

        if model is None or results is None:
            if model is not None:
                try: model.write(os.path.join(optimizer_eg.output_dir, f"failed_main_opt_eg_debug_{start_date.strftime('%Y%m%d')}.lp"), io_options={'symbolic_solver_labels': True})
                except: pass
            return False, None, model, None

        term_cond = results.solver.termination_condition
        solver_status = results.solver.status
        acceptable_conditions = {pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible, pyo.TerminationCondition.locallyOptimal}
        is_successful_solve = term_cond in acceptable_conditions or \
                              (term_cond == pyo.TerminationCondition.maxTimeLimit and solver_status == pyo.SolverStatus.ok) or \
                              (term_cond == pyo.TerminationCondition.other and solver_status == pyo.SolverStatus.ok and term_cond != pyo.TerminationCondition.infeasible)

        if not is_successful_solve:
            return False, None, model, scenario_data

        time_index_for_window = pd.date_range(start=datetime.combine(start_date, datetime.min.time()), periods=horizon_days * 24, freq='H')
        eg_bids_list = optimizer_eg.generate_eg_bids(model, time_index_for_window)

        return True, eg_bids_list, model, scenario_data
    except Exception as e:
        print(f"CRITICAL ERROR in process_stochastic_main_schedule_eg: {e}"); traceback.print_exc()
        return False, None, None, None
