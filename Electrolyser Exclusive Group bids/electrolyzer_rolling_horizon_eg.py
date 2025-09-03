# electrolyzer_rolling_horizon_eg.py exclusive group bids rolling horizon manager


"""
ML-based Rolling Horizon Optimization for an Electrolyzer using Exclusive Group Bids.
This version includes hydrogen storage, a compressor, and minimum-on constraints.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import os
import yaml
import traceback
import time
import matplotlib
matplotlib.use('Agg')

import pyomo.environ as pyo


from synthetic_prices import synth_prices_ar1 # nnot used in the thesis

from ml_forecaster import ElectricityPriceForecaster
from ml_scenario_generator import ScenarioGenerator
from simple_market_clearing_eg import SimpleMarketClearingEG
from stochastic_electrolyzer_bidding_eg import (
    process_stochastic_main_schedule_eg,
    ElectrolyzerStochasticOptimizerEG,
    log_gurobi_iis_eg
)
from electrolyzer_definitions import (
    STORAGE_DEFINITIONS as BASE_STORAGE_DEFINITIONS,
    MATERIALS_IN_STORAGE,
    PLANT_DEFINITIONS,
    TARGET_HYDROGEN_PER_DAY,
    ELECTRICITY_INPUT_MAP,
    ABSOLUTE_PLANT_LIMITS,
    FLEXIBLE_PLANTS
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ElectrolyzerRollingHorizonManagerEG:

    def __init__(self, config_file='config_eg.yaml'):
        print(f"Initializing ElectrolyzerRollingHorizonManagerEG (V5.0 - With Storage) from config: {config_file}")
        try:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            print(f"FATAL ERROR (EG): Could not load or parse config file '{config_file}': {e}"); raise

        sim_params = self.config.get('simulation', {})
        self.forecast_params = self.config.get('forecast', {})
        scenario_params = self.config.get('scenario_generation', {})
        solver_params = self.config.get('solver', {})
        self.electrolyzer_plant_config = self.config.get('electrolyzer_plant', {})
        self.eg_bid_config = self.config.get('exclusive_group_bids', {})
        self.redispatch_config = self.config.get('redispatch', {})
        
        # --- MODIFICATION: Load synthetic data configuration ---
        self.synthetic_config = self.config.get('synthetic_data', {'enabled': False})

        self.baseload_mw = self.electrolyzer_plant_config.get('baseload_mw', 0.0)
        self.start_date = pd.to_datetime(sim_params.get('start_date')).tz_localize(None)
        self.end_date = pd.to_datetime(sim_params.get('end_date')).tz_localize(None)
        self.planning_horizon_days = int(sim_params.get('planning_horizon', 2))
        self.implementation_period_days = int(sim_params.get('implementation_period', 1))
        self.market_input_file = sim_params.get('market_input_file')
        self.output_dir = sim_params.get('output_dir', './electrolyzer_results_v5_storage')
        self.random_seed = int(sim_params.get('random_seed', 42)); np.random.seed(self.random_seed)
        os.makedirs(self.output_dir, exist_ok=True)

        self.forecast_outputs_dir = os.path.join(self.output_dir, 'forecast_outputs_per_window_eg')
        os.makedirs(self.forecast_outputs_dir, exist_ok=True)

        # forecast using synthetic generator or the prophet model
        if not self.synthetic_config.get('enabled', False):
            self.ml_forecaster = ElectricityPriceForecaster(self.forecast_params)
        else:
            self.ml_forecaster = None
            print("INFO: Synthetic data generation is ENABLED. ML forecaster will be bypassed.")

        self.target_variable = self.forecast_params.get('target_variable', 'Price')

        self.scenario_generator = ScenarioGenerator(scenario_params)
        scenario_output_dir_name = f'scenarios_{os.path.basename(self.output_dir.rstrip(os.sep))}'
        scenario_generator_output_dir = os.path.join(self.output_dir, scenario_output_dir_name)
        os.makedirs(scenario_generator_output_dir, exist_ok=True)
        self.scenario_generator.config['output_dir'] = scenario_generator_output_dir

        self.market_clearing_eg = SimpleMarketClearingEG(self.config)
        self.solver_config_for_stochastic = solver_params

        self.optimizer_instance = ElectrolyzerStochasticOptimizerEG(self.config)

        self.current_storage_levels = {mat: BASE_STORAGE_DEFINITIONS[mat]['Initial_Level'] for mat in MATERIALS_IN_STORAGE}
        self.current_plant_operating_levels = {p: ABSOLUTE_PLANT_LIMITS.get(p, {}).get('Min_Op_Abs', 0.0) for p in FLEXIBLE_PLANTS}

        self._reset_results()

        self.target_h2_per_day_benchmark = self.electrolyzer_plant_config.get('target_hydrogen_per_day', TARGET_HYDROGEN_PER_DAY)
        self.total_simulation_days = (self.end_date.date() - self.start_date.date()).days + 1
        self.total_simulation_benchmark_h2_overall = self.target_h2_per_day_benchmark * self.total_simulation_days

        self.simulation_hours_elapsed = 0
        
        print(f"Bidding Strategy: Exclusive Group Bids (V5.0), Num Profiles: {self.eg_bid_config.get('num_profiles_to_generate', 10)}")
        self._load_market_data()

    def _reset_results(self):
        self.results = {
            'accepted_eg_bid_details': [], 'market_clearing_summary_eg': [],
            'plant_operations_details_actual_redispatch': [],
            'forecast_errors': [],
            'hydrogen_produced_periodical_actual': [],
            'redispatch_summary': [],
            'var_values_main_opt': [],
            'cvar_values_main_opt': [],
            'redispatch_unused_energy_periodic_details': [],
            'hourly_site_consumption_actual': [],
            'all_submitted_profiles_detailed': [],
            'economic_analysis': [],
            'plant_energy_consumption_details_expected_main_opt': [],
            'storage_levels_period_end': [],
            'hourly_storage_levels_details_actual_redispatch': [],
            'hourly_storage_levels_details_expected_main_opt': []
        }
        self.total_implemented_h2_actual_so_far = 0.0
        self.total_simulation_unused_energy_mwh = 0.0

    def _load_market_data(self):
        #Use synthetic data generator if enabled in config ---
        if self.synthetic_config.get('enabled', False):
            print("INFO: Generating synthetic market data for the entire simulation period...")
            total_sim_days = (self.end_date - self.start_date).days + self.planning_horizon_days + 1
            
            _, full_actuals_df = self._generate_synthetic_prices(self.start_date, total_sim_days)
            
            self.market_data = full_actuals_df
            print(f"Loaded {len(self.market_data)} rows of synthetic 'actual' prices.")
            return

        # Original code for reading from file, runs if synthetic data is disabled
        try:
            self.market_data = self.ml_forecaster.load_data(self.market_input_file)
            if self.market_data.empty: raise ValueError(f"No data loaded from {self.market_input_file}")
            if pd.api.types.is_datetime64_any_dtype(self.market_data['Date']) and self.market_data['Date'].dt.tz is not None:
                 self.market_data['Date'] = self.market_data['Date'].dt.tz_localize(None)
            print(f"Loaded market data from {self.market_input_file}: {len(self.market_data)} rows, using target '{self.target_variable}'")
        except Exception as e: print(f"ERROR (EG) loading market data: {e}"); raise RuntimeError(f"Failed to load market data") from e

    # New method to call the synthetic generator ---
    def _generate_synthetic_prices(self, start_dt, horizon_days):
        """
        Calls the external synth_prices_ar1 generator with parameters from the config file.
        """
        cfg = self.synthetic_config.copy()
        cfg.pop('enabled', None)

        total_hours = horizon_days * 24

        actual_s, forecast_s = synth_prices_ar1(hours=total_hours, **cfg)

        actual_s.index = pd.date_range(start=start_dt, periods=total_hours, freq='H')
        forecast_s.index = pd.date_range(start=start_dt, periods=total_hours, freq='H')

        forecast_df = pd.DataFrame({'Date': forecast_s.index, 'Forecast_Price': forecast_s.values})
        actuals_df = pd.DataFrame({'Date': actual_s.index, self.target_variable: actual_s.values})

        print(f"  SYNTHETIC DATA: Generated {total_hours}h of prices using synth_prices_ar1.")
        return forecast_df, actuals_df

    def _calculate_forecast_error(self, forecast_df, actual_df_for_period):
        if forecast_df.empty or actual_df_for_period.empty: return {'mape': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}
        fc_df=forecast_df.copy();act_df=actual_df_for_period.copy()
        if not pd.api.types.is_datetime64_any_dtype(fc_df['Date']) or fc_df['Date'].dt.tz is not None: fc_df['Date']=pd.to_datetime(fc_df['Date']).dt.tz_localize(None)
        if not pd.api.types.is_datetime64_any_dtype(act_df['Date']) or act_df['Date'].dt.tz is not None: act_df['Date']=pd.to_datetime(act_df['Date']).dt.tz_localize(None)
        merged=pd.merge(fc_df[['Date','Forecast_Price']], act_df[['Date',self.target_variable]],on='Date',how='inner')
        if merged.empty: return {'mape':np.nan,'rmse':np.nan,'mae':np.nan,'r2':np.nan}
        y_true,y_pred=merged[self.target_variable].values,merged['Forecast_Price'].values
        mape=np.mean(np.abs((y_true-y_pred)/np.maximum(np.abs(y_true),1e-8)))*100
        rmse=np.sqrt(mean_squared_error(y_true,y_pred));mae=mean_absolute_error(y_true,y_pred)
        try: r2=r2_score(y_true,y_pred)
        except: r2=np.nan
        return {'mape':mape,'rmse':rmse,'mae':mae,'r2':r2}

    def _get_tou_weighting_factors_for_timestamp(self, timestamp):
        weekday_weights = {1:[0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.8,0.9,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.9,0.8],2:[0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.8,0.9,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.9,0.8],3:[0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.8,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,1.0,1.0,1.0,1.0,0.9,0.8,0.8],4:[0.7,0.7,0.6,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.7,0.6,0.6,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.8,0.8,0.8,0.8],5:[0.7,0.7,0.6,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.7,0.6,0.6,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.8,0.8,0.8,0.8],6:[0.7,0.7,0.6,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.7,0.6,0.6,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.8,0.8,0.8,0.8],7:[0.7,0.7,0.6,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.7,0.6,0.6,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.8,0.8,0.8,0.8],8:[0.7,0.6,0.7,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.7,0.6,0.6,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.8,0.8,0.8,0.8],9:[0.7,0.6,0.7,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.7,0.6,0.6,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.8,0.8,0.8,0.8],10:[0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.8,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,1.0,1.0,1.0,1.0,1.0,0.9,0.8],11:[0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.8,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,1.0,1.0,1.0,1.0,1.0,0.9,0.8],12:[0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.8,0.9,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.9,0.8]}
        weekend_weights = [0.7,0.6,0.7,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.7,0.6,0.6,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.8,0.8,0.8,0.8]
        try:
            import holidays
            nl_holidays = holidays.country_holidays('NL', years=timestamp.year)
        except ImportError: nl_holidays = {}
        is_weekend = timestamp.dayofweek >= 5
        is_holiday = timestamp.date() in nl_holidays
        if is_weekend or is_holiday: return weekend_weights[timestamp.hour]
        else: return weekday_weights[timestamp.month][timestamp.hour]

    def run_simulation(self):
        print(f"\n===== Starting Electrolyzer EG Rolling Horizon Simulation (V5.0) =====")
        current_dt = self.start_date; period_counter = 1
        while current_dt.date() <= self.end_date.date():
            period_start_time_loop = time.time(); current_date_obj = current_dt.date()
            print(f"\n{'='*80}\nStarting EG optimization period {period_counter}: {current_date_obj.strftime('%Y-%m-%d')}\n{'='*80}")
            print(f"  Current Storage: {{ {', '.join([f'{k}: {v:.2f}' for k,v in self.current_storage_levels.items()])} }}")
            print(f"  Current Plant Ops (entering window): {{ {', '.join([f'{k}: {v:.2f}' for k,v in self.current_plant_operating_levels.items()])} }}")

            days_remaining_in_sim = (self.end_date.date() - current_dt.date()).days + 1
            last_data_date = self.market_data['Date'].max()
            days_available_in_data = (last_data_date.date() - current_dt.date()).days + 1
            days_we_can_plan_for = min(days_remaining_in_sim, days_available_in_data)
            effective_planning_horizon = min(self.planning_horizon_days, days_we_can_plan_for)

            if effective_planning_horizon < self.implementation_period_days:
                print(f"Stopping simulation. Not enough future data for the implementation period starting {current_dt.strftime('%Y-%m-%d')}.")
                break

            forecast_df = pd.DataFrame()
            try:
                # Use synthetic generator or ML forecaster based on config 
                if self.synthetic_config.get('enabled', False):
                    forecast_df, actual_fc_period = self._generate_synthetic_prices(current_dt, effective_planning_horizon)
                else:
                    forecast_df = self.ml_forecaster.forecast(self.market_input_file, current_dt, effective_planning_horizon)
                
                if forecast_df is None or forecast_df.empty: raise ValueError("Forecast generation returned empty.")
                if pd.api.types.is_datetime64_any_dtype(forecast_df['Date']) and forecast_df['Date'].dt.tz is not None: forecast_df['Date'] = forecast_df['Date'].dt.tz_localize(None)
                fc_fname = f"forecast_eg_p{period_counter}_{current_dt.strftime('%Y%m%d')}_h{effective_planning_horizon}d.csv"; forecast_df.to_csv(os.path.join(self.forecast_outputs_dir, fc_fname), index=False)
                impl_end_dt = min(current_dt + timedelta(days=self.implementation_period_days - 1), self.end_date)
                impl_end_ts_fc = impl_end_dt + timedelta(hours=23, minutes=59, seconds=59)
                
                if not self.synthetic_config.get('enabled', False):
                    actual_fc_period = self.market_data.loc[(self.market_data['Date']>=current_dt)&(self.market_data['Date']<=impl_end_ts_fc),['Date',self.target_variable]].copy()
                
                forecast_impl_period = forecast_df.loc[(forecast_df['Date']>=current_dt)&(forecast_df['Date']<=impl_end_ts_fc)].copy()
                fc_error = self._calculate_forecast_error(forecast_impl_period, actual_fc_period)
                self.results['forecast_errors'].append({'period': period_counter, 'start_date': current_date_obj, 'end_date': impl_end_dt.date(), **fc_error})
            except Exception as e: print(f"ERROR (EG) Forecast: {e}"); traceback.print_exc(); break

            try:
                scen_fname = f"scenarios_eg_p{period_counter}_{current_date_obj.strftime('%Y%m%d')}.csv"
                scen_file_opt = os.path.join(self.scenario_generator.config['output_dir'], scen_fname)
                _, scen_file_opt = self.scenario_generator.generate_and_save(forecast_df, current_dt, effective_planning_horizon, output_file=scen_file_opt)
                if not os.path.exists(scen_file_opt): raise FileNotFoundError("Scenario file not created.")
            except Exception as e: print(f"ERROR (EG) Scenario Gen: {e}"); traceback.print_exc(); break

            print(f"\nAttempting MAIN STOCHASTIC EG optimization (V5.0)")
            opt_start_time_main = time.time()
            success_main_opt, submitted_eg_bids_list, main_solved_model, main_scenario_data_cache = \
                process_stochastic_main_schedule_eg(
                    hourly_prices_path=scen_file_opt,
                    start_date=current_date_obj,
                    horizon_days=effective_planning_horizon,
                    output_dir=self.output_dir, config=self.config, solver_config=self.solver_config_for_stochastic,
                    initial_storage_levels=self.current_storage_levels,
                    initial_plant_operating_levels=self.current_plant_operating_levels,
                )
            print(f"Main EG opt (V5.0) processing time: {time.time() - opt_start_time_main:.2f} seconds")

            if success_main_opt and submitted_eg_bids_list:
                baseload_to_add = self.electrolyzer_plant_config.get('baseload_mw', 0.0)
                if baseload_to_add > 0:
                    print(f"  POST-OPTIMIZATION (EG): Adding external baseload of {baseload_to_add:.2f} MW to each hour of each generated profile.")
                    for profile in submitted_eg_bids_list:
                        profile['hourly_mw_values'] = [mw + baseload_to_add for mw in profile['hourly_mw_values']]

            if success_main_opt and main_solved_model is not None and submitted_eg_bids_list is not None:
                print(f"Main Stochastic EG Optimization (V5.0) SUCCEEDED. Generated {len(submitted_eg_bids_list)} profiles.")
                self._process_and_store_eg_results(
                    submitted_eg_bids_list_for_impl_period=submitted_eg_bids_list, actual_prices_df=self.market_data,
                    main_solved_model=main_solved_model, main_scenario_data_cache=main_scenario_data_cache, period_counter=period_counter,
                    period_start_dt=current_dt, period_end_dt_datetime_impl=min(current_dt + timedelta(days=self.implementation_period_days - 1), self.end_date)
                )
            else:
                print(f"Main Stochastic EG Optimization (V5.0) failed for period {period_counter}.")
                self._store_zero_eg_results(period_counter, current_date_obj, min(current_dt + timedelta(days=self.implementation_period_days-1), self.end_date).date(),"Main EG opt failed")
                print(f"  State carried over: Storage: {self.current_storage_levels}, Plant Ops: {self.current_plant_operating_levels}")

            print(f"\n--- Period {period_counter} (EG) processing finished in {time.time() - period_start_time_loop:.2f} seconds ---")

            self.simulation_hours_elapsed += self.implementation_period_days * 24
            current_dt += timedelta(days=self.implementation_period_days); period_counter += 1

        self._save_final_eg_results()
        return self.results

    def _process_and_store_eg_results(self, submitted_eg_bids_list_for_impl_period, actual_prices_df,
                                   main_solved_model, main_scenario_data_cache,
                                   period_counter, period_start_dt, period_end_dt_datetime_impl):
        try:
            hours_in_implementation_period = self.implementation_period_days * 24

            if submitted_eg_bids_list_for_impl_period:
                profiles_to_log = []
                for profile_data in submitted_eg_bids_list_for_impl_period:
                    profile_record = {'period':period_counter,'date':period_start_dt.date(),'profile_id':profile_data.get('profile_id'),'rank':profile_data.get('rank_for_pricing','N/A'),'bid_price':profile_data.get('profile_bid_price'),'total_energy_mwh':profile_data.get('total_energy_mwh')}
                    for i, mw in enumerate(profile_data['hourly_mw_values']): profile_record[f'Hour_{i+1:02d}_MW'] = mw
                    profiles_to_log.append(profile_record)
                self.results['all_submitted_profiles_detailed'].extend(profiles_to_log)

            bids_for_clearing = []
            if submitted_eg_bids_list_for_impl_period:
                for profile_data in submitted_eg_bids_list_for_impl_period:
                    profile_mw_for_clearing = profile_data['hourly_mw_values'][:hours_in_implementation_period]
                    bids_for_clearing.append({'profile_id': profile_data['profile_id'], 'hourly_mw_values': profile_mw_for_clearing, 'profile_bid_price': profile_data['profile_bid_price']})

            accepted_eg_details_df, mc_summary_eg, accepted_hourly_mw_profile = self.market_clearing_eg.clear_market(bids_for_clearing, actual_prices_df, period_start_dt, hours_in_implementation_period)
            self.results['accepted_eg_bid_details'].append(accepted_eg_details_df)
            self.results['market_clearing_summary_eg'].append({'period': period_counter, 'start_date': period_start_dt.date(), **mc_summary_eg})

            self._store_expected_operational_details_eg(main_solved_model, main_scenario_data_cache, period_counter, period_start_dt, hours_in_implementation_period)

            try:
                if main_solved_model and hasattr(main_solved_model, 'VaR'):
                    var_value = pyo.value(main_solved_model.VaR)
                    cvar_value = np.nan
                    if main_scenario_data_cache and hasattr(main_solved_model, 'Excess_Cost_s'):
                        s_probs, c_alpha = main_scenario_data_cache.get('scenario_probs'), main_scenario_data_cache.get('cvar_alpha')
                        if s_probs is not None and c_alpha is not None and (1.0 - c_alpha) > 1e-9:
                            excess_cost = sum(s_probs[s] * pyo.value(main_solved_model.Excess_Cost_s[s]) for s in main_solved_model.S)
                            cvar_value = var_value + (1.0 / (1.0 - c_alpha)) * excess_cost
                    self.results['var_values_main_opt'].append({'period': period_counter, 'VaR_EUR_NetCost': var_value})
                    self.results['cvar_values_main_opt'].append({'period': period_counter, 'CVaR_EUR_NetCost': cvar_value})
                    
                    
                    # This block counts and prints the number of scenarios in the CVaR tail.
                    if hasattr(main_solved_model, 'Excess_Cost_s'):
                        num_total_scenarios = len(main_solved_model.S)
                        worst_case_count = 0
                        # Iterate through each scenario in the solved model
                        for s_index in main_solved_model.S:
                            # If the excess cost for this scenario is positive, it means the scenario's
                            # net cost was higher than the VaR, contributing to the "worst-case" tail.
                            if pyo.value(main_solved_model.Excess_Cost_s[s_index]) > 1e-6: # Use a small tolerance
                                worst_case_count += 1
                        
                        print(f"  CVaR Analysis Insight: {worst_case_count} out of {num_total_scenarios} scenarios are contributing to the worst-case tail (i.e., NetCost > VaR).")
                    

            except Exception:
                self.results['var_values_main_opt'].append({'period': period_counter, 'VaR_EUR_NetCost': np.nan})
                self.results['cvar_values_main_opt'].append({'period': period_counter, 'CVaR_EUR_NetCost': np.nan})

            total_cleared_mwh_impl = mc_summary_eg.get('total_accepted_mw', 0.0); redispatch_successful = False
            if mc_summary_eg.get('accepted_profile_id') is not None and total_cleared_mwh_impl > 1e-3:
                energy_for_flex = [max(0, mw - self.baseload_mw) for mw in accepted_hourly_mw_profile]
                optimizer_rd = self.optimizer_instance
                redispatch_model, rd_results = optimizer_rd.optimize_energy_constrained_schedule_eg(
                    start_datetime_obj=period_start_dt, num_hours_to_redispatch=hours_in_implementation_period,
                    initial_storage_levels_for_redispatch=self.current_storage_levels,
                    initial_plant_operating_levels_for_redispatch=self.current_plant_operating_levels,
                    hourly_cleared_energy_profile=energy_for_flex
                )
                if rd_results and rd_results.solver.termination_condition in [pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible, pyo.TerminationCondition.locallyOptimal]:
                    redispatch_successful = True
                    self._store_actual_details_and_update_state_eg(redispatch_model, "actual_from_eg_redispatch", period_counter, period_start_dt, hours_in_implementation_period, period_end_dt_datetime_impl.date())
                else:
                    self._store_failed_operation_actuals_and_state_eg(period_counter, period_start_dt, hours_in_implementation_period, period_end_dt_datetime_impl.date(), "EG Re-dispatch Failed")
            else:
                self._store_failed_operation_actuals_and_state_eg(period_counter, period_start_dt, hours_in_implementation_period, period_end_dt_datetime_impl.date(), "No EG Profile Cleared")

            self.results['redispatch_summary'].append({'period': period_counter, 'success': redispatch_successful})

        except Exception as e:
            print(f"ERROR (EG) processing results period {period_counter}: {e}"); traceback.print_exc()

    def _store_expected_operational_details_eg(self, main_solved_model, main_scenario_data_cache, period_counter, period_start_dt, hours_in_implementation):
        if main_solved_model is None or main_scenario_data_cache is None: return
        s_probs = main_scenario_data_cache.get('scenario_probs')
        if s_probs is None: return

        detail_plant_consump_exp, detail_hrly_storage_exp = [], []
        time_index_for_impl_period = pd.date_range(start=period_start_dt, periods=hours_in_implementation, freq='H')

        for h_idx in range(hours_in_implementation):
            ts = time_index_for_impl_period[h_idx]
            for p_name in FLEXIBLE_PLANTS:
                exp_p_val = sum(s_probs[s_idx] * pyo.value(main_solved_model.P_plant_s[s_idx, p_name, h_idx]) for s_idx in main_solved_model.S)
                elec_f = ELECTRICITY_INPUT_MAP.get(p_name, 0.0)
                abs_elec = abs(exp_p_val * elec_f)
                detail_plant_consump_exp.append({'Period': period_counter, 'Timestamp': ts, 'Plant_Name': p_name, 'Expected_Operation_Level': round(exp_p_val, 3), 'Expected_Absolute_Electricity_MWh': round(abs_elec, 3)})
            
            for mat_st in MATERIALS_IN_STORAGE:
                exp_st_lvl = sum(s_probs[s_idx] * pyo.value(main_solved_model.S_material_s[s_idx, mat_st, h_idx + 1]) for s_idx in main_solved_model.S)
                detail_hrly_storage_exp.append({'Period': period_counter, 'Timestamp': ts, 'Material': mat_st, 'Expected_Storage_Level_Tons': round(exp_st_lvl, 2)})

        self.results['plant_energy_consumption_details_expected_main_opt'].extend(detail_plant_consump_exp)
        self.results['hourly_storage_levels_details_expected_main_opt'].extend(detail_hrly_storage_exp)


    def _store_actual_details_and_update_state_eg(self, model_to_use, model_type, period_counter, period_start_dt, hours_in_implementation, period_end_dt_date):
        act_h2_total=0.0
        plant_ops_act, hrly_storage_act, new_storage, new_plant_ops = [],[],{},{}
        time_idx_impl = pd.date_range(start=period_start_dt, periods=hours_in_implementation, freq='H')

        if model_to_use is None:
            self._store_failed_operation_actuals_and_state_eg(period_counter,period_start_dt,hours_in_implementation,period_end_dt_date,"Model None"); return

        total_unused_energy_this_period = sum(pyo.value(model_to_use.Unused_Cleared_Energy_t[t]) for t in model_to_use.H)
        self.total_simulation_unused_energy_mwh += total_unused_energy_this_period
        self.results['redispatch_unused_energy_periodic_details'].append({'period': period_counter, 'total_unused_mwh': total_unused_energy_this_period})

        for h_idx in range(hours_in_implementation):
            ts = time_idx_impl[h_idx]
            total_flex_elec_this_hour = 0
            if hasattr(model_to_use, 'P_plant'):
                act_h2_total += pyo.value(model_to_use.P_plant['Compressor', h_idx])

            for p_name in FLEXIBLE_PLANTS:
                p_op = pyo.value(model_to_use.P_plant[p_name,h_idx])
                abs_elec = abs(p_op * ELECTRICITY_INPUT_MAP.get(p_name, 0.0))
                total_flex_elec_this_hour += abs_elec
                plant_ops_act.append({'Period':period_counter,'Timestamp':ts,'Plant_Name':p_name,'Actual_Operation_Level':round(p_op, 3),'Actual_Absolute_Electricity_MWh':round(abs_elec,3),'Source_Model_Type':model_type})
                if h_idx==hours_in_implementation-1: new_plant_ops[p_name]=p_op

            total_site_consumption = total_flex_elec_this_hour + self.baseload_mw
            self.results['hourly_site_consumption_actual'].append({'Timestamp': ts, 'Total_Site_Consumption_MWh': total_site_consumption})
            
            for mat in MATERIALS_IN_STORAGE:
                s_lvl=pyo.value(model_to_use.S_material[mat,h_idx+1])
                hrly_storage_act.append({'Period':period_counter,'Timestamp':ts,'Material':mat,'Actual_Storage_Level_Tons':round(s_lvl,2)})
                if h_idx==hours_in_implementation-1: new_storage[mat]=max(0.0,s_lvl)

        self.results['plant_operations_details_actual_redispatch'].extend(plant_ops_act)
        self.results['hourly_storage_levels_details_actual_redispatch'].extend(hrly_storage_act)
        self.total_implemented_h2_actual_so_far+=act_h2_total
        self.results['hydrogen_produced_periodical_actual'].append({'period':period_counter,'start_date':period_start_dt.date(),'end_date':period_end_dt_date,'total_actual_hydrogen_kg':act_h2_total})
        
        if new_storage: self.current_storage_levels=new_storage
        if new_plant_ops: self.current_plant_operating_levels=new_plant_ops
        self.results['storage_levels_period_end'].append({'period':period_counter,'date':period_end_dt_date.strftime('%Y-%m-%d'),'type':model_type,**self.current_storage_levels})


    def _store_failed_operation_actuals_and_state_eg(self, period_counter, period_start_dt, hours_in_implementation, period_end_dt_date, reason="Op Failed"):
        print(f"  Storing ZERO actuals for EG period {period_counter} due to: {reason}. State carries over.")
        time_idx_impl = pd.date_range(start=period_start_dt, periods=hours_in_implementation, freq='H')
        self.results['hydrogen_produced_periodical_actual'].append({'period':period_counter,'start_date':period_start_dt.date(),'end_date':period_end_dt_date,'total_actual_hydrogen_kg':0.0})
        for ts in time_idx_impl:
            self.results['hourly_site_consumption_actual'].append({'Timestamp': ts, 'Total_Site_Consumption_MWh': self.baseload_mw})
            for p_name in FLEXIBLE_PLANTS: self.results['plant_operations_details_actual_redispatch'].append({'Period':period_counter,'Timestamp':ts,'Plant_Name':p_name,'Actual_Operation_Level':0.0,'Actual_Absolute_Electricity_MWh':0.0})
        self.results['storage_levels_period_end'].append({'period':period_counter,'date':period_end_dt_date.strftime('%Y-%m-%d'),'comment':f'{reason}. State carried over.',**self.current_storage_levels})

    def _store_zero_eg_results(self, period_counter, current_date_obj, period_end_date_obj, reason="Opt failed"):
        print(f"Storing ZERO EG results for period {period_counter}. Reason: {reason}")
        self.results['market_clearing_summary_eg'].append({'period':period_counter,'start_date':current_date_obj,'end_date':period_end_date_obj,'total_accepted_mw':0.0,'total_cost_cleared':0.0,'avg_price_cleared':0.0, 'accepted_profile_id':None,'comment':reason})
        self._store_failed_operation_actuals_and_state_eg(period_counter, datetime.combine(current_date_obj,datetime.min.time()), self.implementation_period_days*24, period_end_date_obj, reason)
        self.results['var_values_main_opt'].append({'period': period_counter, 'VaR_EUR_NetCost': np.nan})
        self.results['cvar_values_main_opt'].append({'period': period_counter, 'CVaR_EUR_NetCost': np.nan})

    def _calculate_and_store_economic_indicators_eg(self):
        print("\nCalculating final EG economic and operational indicators for Electrolyzer...")
        accepted_df = pd.concat(self.results['accepted_eg_bid_details'], ignore_index=True) if self.results['accepted_eg_bid_details'] else pd.DataFrame()
        h2_actual_df = pd.DataFrame(self.results['hydrogen_produced_periodical_actual'])
        ops_df = pd.DataFrame(self.results['plant_operations_details_actual_redispatch'])
        var_df = pd.DataFrame(self.results['var_values_main_opt'])
        cvar_df = pd.DataFrame(self.results['cvar_values_main_opt'])
        indicators = {}
        final_tou_cost = 0.0
        tou_config = self.config.get('time_of_use_tariff', {})
        if tou_config.get('enabled', False) and self.results['hourly_site_consumption_actual']:
            consumption_df = pd.DataFrame(self.results['hourly_site_consumption_actual'])
            consumption_df['Timestamp'] = pd.to_datetime(consumption_df['Timestamp'])
            consumption_df['weight'] = consumption_df['Timestamp'].apply(self._get_tou_weighting_factors_for_timestamp)
            consumption_df['weighted_consumption_mw'] = consumption_df['Total_Site_Consumption_MWh'] * consumption_df['weight']
            simulation_peak_mw = consumption_df['weighted_consumption_mw'].max()
            tou_rate = tou_config.get('rate_eur_per_kw_month', 0.0)
            final_tou_cost = simulation_peak_mw * 1000 * tou_rate
            indicators['TOU - Simulation Weighted Peak'] = (simulation_peak_mw, 'MW')
            indicators['TOU - Final Peak Charge'] = (final_tou_cost, 'EUR')
        if not accepted_df.empty and not h2_actual_df.empty:
            total_mwh_consumed = accepted_df['Accepted MW'].sum()
            total_market_cost = accepted_df['Cost'].sum()
            total_electricity_cost = total_market_cost + final_tou_cost
            indicators['Total MWh Consumed'] = (total_mwh_consumed, 'MWh')
            indicators['Total Electricity Cost (Market + TOU)'] = (total_electricity_cost, 'EUR')
            indicators['Average Electricity Price Paid (incl. TOU)'] = (total_electricity_cost / total_mwh_consumed if total_mwh_consumed > 1e-6 else 0, 'EUR/MWh')
            if 'MCP' in accepted_df.columns:
                hourly_prices = accepted_df.groupby('Date')['MCP'].first()
                indicators['Std Dev of Hourly Price Paid (Risk)'] = (hourly_prices.std(), 'EUR/MWh')
                indicators['Average Market Price (Benchmark)'] = (hourly_prices.mean(), 'EUR/MWh')
            total_h2 = h2_actual_df['total_actual_hydrogen_kg'].sum()
            indicators['Total Hydrogen Produced (Compressed)'] = (total_h2, 'kg')
            indicators['Electricity Consumption per kg H2'] = (total_mwh_consumed * 1000 / total_h2 if total_h2 > 1e-6 else 0, 'kWh/kg')
            indicators['Electricity Cost per kg H2'] = (total_electricity_cost / total_h2 if total_h2 > 1e-6 else 0, 'EUR/kg')
            h2_price_config = self.config.get('hydrogen_price', {})
            h2_price_per_kg = float(h2_price_config.get('value_eur_per_kg', 0.0))
            total_revenue_from_h2 = total_h2 * h2_price_per_kg
            net_profit = total_revenue_from_h2 - total_electricity_cost
            profit_margin_percentage = (net_profit / total_revenue_from_h2 * 100) if total_revenue_from_h2 > 1e-6 else 0.0
            indicators.update({'Total Revenue from Hydrogen':(total_revenue_from_h2, 'EUR'), 'Net Profit':(net_profit, 'EUR'), 'Profit Margin':(profit_margin_percentage, '%')})
        if not var_df.empty: indicators['Average VaR (NetCost)'] = (var_df['VaR_EUR_NetCost'].mean(), 'EUR')
        if not cvar_df.empty: indicators['Average CVaR (NetCost)'] = (cvar_df['CVaR_EUR_NetCost'].mean(), 'EUR')
        analysis_data = [{'Indicator': key, 'Value': val, 'Unit': unit} for key, (val, unit) in indicators.items()]
        self.results['economic_analysis'] = pd.DataFrame(analysis_data)
        print("  Economic and operational indicators for EG calculated successfully.")

    def _save_final_eg_results(self):
        output_prefix = os.path.basename(self.output_dir.rstrip(os.sep))
        if not output_prefix: output_prefix = f"electrolyzer_sim_eg_results_{self.start_date.strftime('%Y%m%d')}"
        self._calculate_and_store_economic_indicators_eg()
        
        # Save the synthetic actual prices if they were used ---
        if self.synthetic_config.get('enabled', False):
            print("\n--- SAVING SYNTHETIC ACTUAL PRICES (Electrolyzer EG) ---")
            try:
                end_of_sim_dt = self.end_date + timedelta(days=1) - timedelta(seconds=1)
                sim_actuals_df = self.market_data[
                    (self.market_data['Date'] >= self.start_date) &
                    (self.market_data['Date'] <= end_of_sim_dt)
                ].copy()

                sim_actuals_df.rename(columns={self.target_variable: 'Synthetic_Actual_MCP'}, inplace=True)
                
                fpath_actuals = os.path.join(self.output_dir, f"{output_prefix}_synthetic_actual_prices.xlsx")
                sim_actuals_df.to_excel(fpath_actuals, index=False)
                print(f"Saved synthetic actual prices to {fpath_actuals}")

            except Exception as e_save_synth:
                print(f"Error saving synthetic actual prices: {e_save_synth}")
     

        print(f"\n\n{'='*30} FINAL EG SIMULATION SUMMARY (V5.0) {'='*30}")
        if self.results['economic_analysis'] is not None and not self.results['economic_analysis'].empty:
            print(self.results['economic_analysis'].to_string(index=False))
        print(f"\n--- FINAL ACTUAL PRODUCTION SUMMARY (EG) ---")
        print(f"  Target H2 Benchmark: {self.total_simulation_benchmark_h2_overall:.2f} kg. Actual H2: {self.total_implemented_h2_actual_so_far:.2f} kg.")
        shortfall = self.total_simulation_benchmark_h2_overall - self.total_implemented_h2_actual_so_far
        if shortfall > 1e-3: print(f"  H2 Shortfall: {shortfall:.2f} kg")
        else: print(f"  H2 Surplus: {-shortfall:.2f} kg")
        print(f"\n--- TOTAL UNUSED CLEARED ENERGY (SIMULATION TOTAL EG) ---")
        print(f"  Total Unused: {self.total_simulation_unused_energy_mwh:.3f} MWh")
        for key,data in self.results.items():
            if (isinstance(data, list) or isinstance(data, dict)) and not data: continue
            df_save=None
            try:
                if isinstance(data, list):
                    if data and isinstance(data[0], dict): df_save=pd.DataFrame(data)
                    elif data and isinstance(data[0], pd.DataFrame): df_save=pd.concat(data, ignore_index=True)
                elif isinstance(data, dict):
                    if data: df_save = pd.DataFrame.from_dict(data, orient='index', columns=['Value']).reset_index().rename(columns={'index':'Material'})
                elif isinstance(data, pd.DataFrame): df_save = data
            except (IndexError, TypeError): continue
            if df_save is not None and not df_save.empty:
                try:
                    fpath=os.path.join(self.output_dir,f"{output_prefix}_{key}.xlsx")
                    df_save.to_excel(fpath,index=False)
                    print(f"Saved {key} (EG) to {fpath}")
                except Exception as e_save:
                    print(f"Error saving {key} (EG) to Excel: {e_save}")
        print("\n" + "="*28 + " END OF FINAL EG SIMULATION SUMMARY " + "="*27 + "\n")
