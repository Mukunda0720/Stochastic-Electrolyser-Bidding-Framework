# electrolyzer_rolling_horizon.py
#rolling horizon manager

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


from synthetic_prices import synth_prices_ar1 # was used only for tata steel case but works the same here

from ml_forecaster import ElectricityPriceForecaster
from ml_scenario_generator import ScenarioGenerator
from simple_market_clearing import SimpleMarketClearing
from stochastic_electrolyzer_bidding import (
    process_stochastic_main_schedule_must_run, 
    ElectrolyzerStochasticOptimizer,
    log_gurobi_iis 
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


class ElectrolyzerRollingHorizonManager:

    def __init__(self, config_file='config.yaml'):
        print(f"Initializing ElectrolyzerRollingHorizonManager (V5.2 - Full System Model with Expected Outputs) from config: {config_file}")
        try:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            print(f"FATAL ERROR: Could not load or parse config file '{config_file}': {e}"); raise

        sim_params = self.config.get('simulation', {})
        self.forecast_params = self.config.get('forecast', {})
        scenario_params = self.config.get('scenario_generation', {})
        solver_params = self.config.get('solver', {})
        self.electrolyzer_plant_config = self.config.get('electrolyzer_plant', {}) 
        self.redispatch_config = self.config.get('redispatch', {})
        self.bid_strategy_config = self.config.get('bid_strategy', {})
        

        self.synthetic_config = self.config.get('synthetic_data', {'enabled': False})

        self.baseload_mw = self.electrolyzer_plant_config.get('baseload_mw', 0.0) 
        self.start_date = pd.to_datetime(sim_params.get('start_date')).tz_localize(None)
        self.end_date = pd.to_datetime(sim_params.get('end_date')).tz_localize(None)
        self.planning_horizon = int(sim_params.get('planning_horizon', 2))
        self.implementation_period = int(sim_params.get('implementation_period', 1))
        self.market_input_file = sim_params.get('market_input_file')
        self.output_dir = sim_params.get('output_dir', './electrolyzer_results_v5_hourly') 
        self.random_seed = int(sim_params.get('random_seed', 42)); np.random.seed(self.random_seed)
        os.makedirs(self.output_dir, exist_ok=True)

        self.forecast_outputs_dir = os.path.join(self.output_dir, 'forecast_outputs_per_window')
        os.makedirs(self.forecast_outputs_dir, exist_ok=True)

        # initialize the ML forecaster ---
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

        self.market_clearing = SimpleMarketClearing({'output_dir': self.output_dir})
        self.solver_config_for_stochastic = solver_params 
        self.optimizer_instance = ElectrolyzerStochasticOptimizer(self.config)

        self.current_storage_levels = {mat: BASE_STORAGE_DEFINITIONS[mat]['Initial_Level'] for mat in MATERIALS_IN_STORAGE}
        self.current_plant_operating_levels = {p: ABSOLUTE_PLANT_LIMITS.get(p, {}).get('Min_Op_Abs', 0.0) for p in FLEXIBLE_PLANTS}
        
        self._reset_results() 

        self.target_hydrogen_per_day_benchmark = self.electrolyzer_plant_config.get('target_hydrogen_per_day', TARGET_HYDROGEN_PER_DAY)
        self.total_simulation_days = (self.end_date.date() - self.start_date.date()).days + 1
        self.total_simulation_benchmark_h2_overall = self.target_hydrogen_per_day_benchmark * self.total_simulation_days
        
        print(f"Bidding Strategy: Hourly Bids (V5.2 - Full System Model with Expected Outputs)")
        self._load_market_data()

    def _reset_results(self): 
        self.results = {
            'bids': [], 'accepted': [],
            'plant_energy_consumption_details_expected_main_opt': [],
            'hourly_storage_levels_details_expected_main_opt': [],
            'hydrogen_produced_periodical_expected_main_opt': [],
            'plant_operations_details_actual_redispatch': [], 
            'storage_levels_period_end': [],
            'hourly_storage_levels_details_actual_redispatch': [],
            'costs': [], 'forecast_errors': [],
            'hydrogen_produced_periodical_actual': [], 
            'redispatch_summary': [], 'var_values_main_opt': [], 'cvar_values_main_opt': [],
            'redispatch_unused_energy_periodic_details': [],
            'hourly_site_consumption_actual': [],
            'economic_analysis': []
        }
        self.total_implemented_h2_actual_so_far = 0.0
        self.total_simulation_unused_energy_mwh = 0.0
        self.current_storage_levels = {mat: BASE_STORAGE_DEFINITIONS[mat]['Initial_Level'] for mat in MATERIALS_IN_STORAGE}

    def _load_market_data(self):
        if self.synthetic_config.get('enabled', False):
            print("INFO: Generating synthetic market data for the entire simulation period...")
            total_sim_days = (self.end_date - self.start_date).days + self.planning_horizon + 1
            _, full_actuals_df = self._generate_synthetic_prices(self.start_date, total_sim_days)
            self.market_data = full_actuals_df
            print(f"Loaded {len(self.market_data)} rows of synthetic 'actual' prices.")
            return

        try:
            self.market_data = self.ml_forecaster.load_data(self.market_input_file)
            if self.market_data.empty: raise ValueError(f"No data loaded from {self.market_input_file}")
            if pd.api.types.is_datetime64_any_dtype(self.market_data['Date']) and self.market_data['Date'].dt.tz is not None:
                 self.market_data['Date'] = self.market_data['Date'].dt.tz_localize(None)
            print(f"Loaded market data from {self.market_input_file}: {len(self.market_data)} rows, using target '{self.target_variable}'")
        except Exception as e: print(f"ERROR loading market data: {e}"); raise RuntimeError(f"Failed to load market data") from e

    def _generate_synthetic_prices(self, start_dt, horizon_days):
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

    def _calculate_forecast_error(self, forecast_df, actual_df): 
        if not pd.api.types.is_datetime64_any_dtype(forecast_df['Date']) or forecast_df['Date'].dt.tz is not None: forecast_df['Date'] = pd.to_datetime(forecast_df['Date']).dt.tz_localize(None)
        if not pd.api.types.is_datetime64_any_dtype(actual_df['Date']) or actual_df['Date'].dt.tz is not None: actual_df['Date'] = pd.to_datetime(actual_df['Date']).dt.tz_localize(None)
        merged = pd.merge(forecast_df[['Date', 'Forecast_Price']], actual_df[['Date', self.target_variable]], on='Date', how='inner')
        if merged.empty: return {'mape': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}
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

    def _generate_dynamic_bid_prices_electrolyzer(self, scenario_prices_matrix, total_hours_in_window):
        dynamic_bid_prices = {}
        opportunity_cost_mwh_shortfall = self.electrolyzer_plant_config.get('opportunity_cost_mwh_shortfall', 1000.0)
        general_market_price_cap = self.forecast_params.get('prophet', {}).get('cap', 4000.0)
        for t_idx in range(total_hours_in_window):
            current_hour_prices = list(scenario_prices_matrix[:, t_idx])
            current_hour_prices.extend([opportunity_cost_mwh_shortfall, general_market_price_cap])
            final_prices_temp = sorted(list(set(current_hour_prices)), reverse=True)
            final_prices = [max(0.0, p) for p in final_prices_temp]
            max_b_config = self.bid_strategy_config.get('max_blocks_per_hour', 18)
            if len(final_prices) > max_b_config:
                final_prices = final_prices[:max_b_config]
            dynamic_bid_prices[t_idx] = [round(p, 2) for p in final_prices] if final_prices else [0.0]
        return dynamic_bid_prices

    def run_simulation(self):
        print(f"\n===== Starting Electrolyzer Rolling Horizon Simulation (V5.2 - Full System with Expected Outputs) =====")
        current_dt = self.start_date
        period_counter = 1
        
        while current_dt.date() <= self.end_date.date():
            period_start_time = time.time()
            current_date_obj = current_dt.date()
            print(f"\n{'='*80}\nStarting optimization period {period_counter}: {current_date_obj.strftime('%Y-%m-%d')}\n{'='*80}")
            print(f"  Current Storage: {{ {', '.join([f'{k}: {v:.2f}' for k,v in self.current_storage_levels.items()])} }}")
            print(f"  Current Plant Ops (entering window): {{ {', '.join([f'{k}: {v:.2f}' for k,v in self.current_plant_operating_levels.items()])} }}")
            
            days_remaining_in_sim = (self.end_date.date() - current_dt.date()).days + 1
            last_data_date = self.market_data['Date'].max()
            days_available_in_data = (last_data_date.date() - current_dt.date()).days + 1
            days_we_can_plan_for = min(days_remaining_in_sim, days_available_in_data)
            days_in_horizon_window = min(self.planning_horizon, days_we_can_plan_for)

            if days_in_horizon_window < self.implementation_period:
                print(f"Stopping simulation. Not enough future data for implementation period.")
                break

            try:
                if self.synthetic_config.get('enabled', False):
                    forecast_df, actual_prices_fc = self._generate_synthetic_prices(current_dt, days_in_horizon_window)
                else:
                    forecast_df = self.ml_forecaster.forecast(self.market_input_file, current_dt, days_in_horizon_window)
                    impl_end_dt_fc = min(current_dt + timedelta(days=self.implementation_period - 1), self.end_date)
                    impl_end_ts_fc = impl_end_dt_fc + timedelta(hours=23, minutes=59, seconds=59)
                    actual_prices_fc = self.market_data.loc[(self.market_data['Date'] >= current_dt) & (self.market_data['Date'] <= impl_end_ts_fc), ['Date', self.target_variable]].copy()

                # forecast file save 
                # This block saves the generated forecast to a file, which the plotting script needs.
                forecast_filename = f"forecast_p{period_counter}_{current_dt.strftime('%Y%m%d')}_h{days_in_horizon_window}d.csv"
                forecast_filepath = os.path.join(self.forecast_outputs_dir, forecast_filename)
                if forecast_df is not None and not forecast_df.empty:
                    forecast_df.to_csv(forecast_filepath, index=False)
                    print(f"  Saved forecast for period {period_counter} to {forecast_filepath}")


                scen_fname = f"scenarios_p{period_counter}_{current_dt.strftime('%Y%m%d')}.csv"
                scen_file_opt = os.path.join(self.scenario_generator.config['output_dir'], scen_fname) 
                _, scen_file_opt = self.scenario_generator.generate_and_save(forecast_df, current_dt, days_in_horizon_window, output_file=scen_file_opt)
                scen_df_bids = pd.read_csv(scen_file_opt)
                price_cols = [col for col in scen_df_bids.columns if col.startswith('Hour_')]
                raw_scen_prices_matrix = scen_df_bids[price_cols].values
                dyn_bid_prices_window = self._generate_dynamic_bid_prices_electrolyzer(raw_scen_prices_matrix, days_in_horizon_window * 24)
                num_blocks_hr = {t: len(prices) for t, prices in dyn_bid_prices_window.items()}


                try:
                    # Determine the actual prices for the implementation period
                    impl_end_dt = current_dt + timedelta(days=self.implementation_period - 1, hours=23)
                    actuals_for_period = self.market_data[
                        (self.market_data['Date'] >= current_dt) & (self.market_data['Date'] <= impl_end_dt)
                    ].copy()
                    
                    # Align forecast with the implementation period
                    forecast_for_period = forecast_df[forecast_df['Date'].isin(actuals_for_period['Date'])].copy()

                    if not forecast_for_period.empty and not actuals_for_period.empty:
                        error_metrics = self._calculate_forecast_error(forecast_for_period, actuals_for_period)
                        self.results['forecast_errors'].append({
                            'period': period_counter,
                            'start_date': current_date_obj,
                            **error_metrics
                        })
                        print(f"  Successfully calculated and stored forecast errors for period {period_counter}.")
                except Exception as e_fe:
                    print(f"  WARNING: Could not calculate forecast error for period {period_counter}: {e_fe}")


            except Exception as e: print(f"ERROR: Forecast/Scenario/Bid Price generation failed: {e}"); traceback.print_exc(); break

            print(f"\nAttempting MAIN STOCHASTIC optimization (V5.2)")
            opt_start_time_main = time.time()
            success_main_opt, bids_df, main_solved_model, main_scenario_data_cache = \
                process_stochastic_main_schedule_must_run( 
                    hourly_prices_path=scen_file_opt, start_date=current_date_obj, horizon_days=days_in_horizon_window, 
                    output_dir=self.output_dir, config=self.config, solver_config=self.solver_config_for_stochastic, 
                    initial_storage_levels=self.current_storage_levels,
                    initial_plant_operating_levels=self.current_plant_operating_levels, 
                    dynamic_bid_prices_for_tiers=dyn_bid_prices_window, num_blocks_per_hour=num_blocks_hr
                )
            print(f"Main opt (V5.2) processing time: {time.time() - opt_start_time_main:.2f} seconds")

            if success_main_opt and main_solved_model is not None:
                self._process_and_store_results(bids_df, self.market_data, main_solved_model, 
                    main_scenario_data_cache, period_counter, current_dt)
            else:
                 self._store_zero_results(period_counter, current_date_obj, "Main opt failed")

            print(f"\n--- Period {period_counter} processing finished in {time.time() - period_start_time:.2f} seconds ---")
            
            current_dt += timedelta(days=self.implementation_period)
            period_counter += 1
        
        self._save_final_results()
        return self.results

    def _process_and_store_results(self, bids_df, actual_prices_df, main_solved_model,
                                   main_scenario_data_cache, period_counter, period_start_dt):
        try:
            hours_in_implementation = self.implementation_period * 24
            time_index_for_impl_period = pd.date_range(start=period_start_dt, periods=hours_in_implementation, freq='H')
            period_end_dt_datetime = time_index_for_impl_period[-1]
            period_end_dt_date = period_end_dt_datetime.date()

            impl_bids_df = pd.DataFrame() 
            if bids_df is not None and not bids_df.empty:
                impl_mask = (bids_df['Date'] >= period_start_dt) & (bids_df['Date'] <= period_end_dt_datetime)
                impl_bids_df = bids_df.loc[impl_mask].copy()
            
            accepted_bids_impl_df = self.market_clearing.clear_market(impl_bids_df, actual_prices_df)
            summary_mc = self.market_clearing.calculate_summary(accepted_bids_impl_df)
            
            self.results['bids'].append(impl_bids_df)
            self.results['accepted'].append(accepted_bids_impl_df)
            self.results['costs'].append({'period':period_counter, **summary_mc})
            
            self._store_expected_operational_details(main_solved_model, main_scenario_data_cache, period_counter, time_index_for_impl_period)
            
            try:
                if main_solved_model and hasattr(main_solved_model, 'VaR'):
                    var_value = pyo.value(main_solved_model.VaR, exception=False)
                    cvar_value = np.nan
                    if main_scenario_data_cache and hasattr(main_solved_model, 'Excess_Cost_s'):
                        s_probs, c_alpha = main_scenario_data_cache.get('scenario_probs'), main_scenario_data_cache.get('cvar_alpha')
                        if s_probs is not None and c_alpha is not None and (1.0 - c_alpha) > 1e-9:
                            excess_cost = sum(s_probs[s] * pyo.value(main_solved_model.Excess_Cost_s[s]) for s in main_solved_model.S)
                            cvar_value = var_value + (1.0 / (1.0 - c_alpha)) * excess_cost
                    self.results['var_values_main_opt'].append({'period': period_counter, 'VaR_EUR_NetCost': var_value})
                    self.results['cvar_values_main_opt'].append({'period': period_counter, 'CVaR_EUR_NetCost': cvar_value})
            except Exception:
                self.results['var_values_main_opt'].append({'period': period_counter, 'VaR_EUR_NetCost': np.nan})
                self.results['cvar_values_main_opt'].append({'period': period_counter, 'CVaR_EUR_NetCost': np.nan})

            total_cleared_mwh_impl = summary_mc.get('total_accepted_mw', 0.0)
            redispatch_successful = False
            if total_cleared_mwh_impl > 1e-3:
                accepted_energy_profile = accepted_bids_impl_df.groupby('Date')['Accepted MW'].sum().reindex(time_index_for_impl_period, fill_value=0).tolist()
                energy_for_flex = [max(0, mw - self.baseload_mw) for mw in accepted_energy_profile]
                
                optimizer_rd = self.optimizer_instance
                redispatch_model, rd_results = optimizer_rd.optimize_energy_constrained_schedule(
                    start_datetime_obj=period_start_dt, num_hours_to_redispatch=hours_in_implementation,
                    initial_storage_levels_for_redispatch=self.current_storage_levels,
                    initial_plant_operating_levels_for_redispatch=self.current_plant_operating_levels,
                    hourly_energy_constraints=energy_for_flex
                )
                if rd_results and rd_results.solver.termination_condition in [pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible, pyo.TerminationCondition.locallyOptimal]:
                    redispatch_successful = True
                    self._store_actual_details_and_update_state(redispatch_model, "actual_from_redispatch", period_counter, time_index_for_impl_period, period_end_dt_date)
                else:
                    self._store_failed_operation_actuals_and_state(period_counter, period_start_dt, hours_in_implementation, period_end_dt_date, "Re-dispatch Failed")
            else:
                self._store_failed_operation_actuals_and_state(period_counter, period_start_dt, hours_in_implementation, period_end_dt_date, "No Energy Cleared")
            
            self.results['redispatch_summary'].append({'period': period_counter, 'success': redispatch_successful})

        except Exception as e: print(f"ERROR processing results for period {period_counter}: {e}"); traceback.print_exc() 

    def _store_expected_operational_details(self, main_solved_model, main_scenario_data_cache, period_counter, time_index_for_impl_period):
        """Calculates and stores the probability-weighted average results from the main optimization."""
        if main_solved_model is None or main_scenario_data_cache is None:
            print("  Skipping expected results: main model or scenario data is missing.")
            return

        s_probs = main_scenario_data_cache.get('scenario_probs')
        if s_probs is None:
            print("  Skipping expected results: scenario probabilities are missing.")
            return

        hours_in_implementation = len(time_index_for_impl_period)
        detail_plant_consump_exp = []
        detail_hrly_storage_exp = []
        expected_h2_total = 0.0

        for h_idx in range(hours_in_implementation):
            ts = time_index_for_impl_period[h_idx]
            
            if hasattr(main_solved_model, 'P_plant_s'):
                expected_h2_total += sum(s_probs[s_idx] * pyo.value(main_solved_model.P_plant_s[s_idx, 'Compressor', h_idx]) for s_idx in main_solved_model.S)

            for p_name in FLEXIBLE_PLANTS:
                if hasattr(main_solved_model, 'P_plant_s'):
                    exp_p_val = sum(s_probs[s_idx] * pyo.value(main_solved_model.P_plant_s[s_idx, p_name, h_idx]) for s_idx in main_solved_model.S)
                    elec_f = ELECTRICITY_INPUT_MAP.get(p_name, 0.0)
                    abs_elec = abs(exp_p_val * elec_f)
                    detail_plant_consump_exp.append({
                        'Period': period_counter, 
                        'Timestamp': ts, 
                        'Plant_Name': p_name, 
                        'Expected_Operation_Level': round(exp_p_val, 3), 
                        'Expected_Absolute_Electricity_MWh': round(abs_elec, 3)
                    })
            
            for mat_st in MATERIALS_IN_STORAGE:
                if hasattr(main_solved_model, 'S_material_s'):
                    exp_st_lvl = sum(s_probs[s_idx] * pyo.value(main_solved_model.S_material_s[s_idx, mat_st, h_idx + 1]) for s_idx in main_solved_model.S)
                    detail_hrly_storage_exp.append({
                        'Period': period_counter, 
                        'Timestamp': ts, 
                        'Material': mat_st, 
                        'Expected_Storage_Level_Tons': round(exp_st_lvl, 2)
                    })

        self.results['plant_energy_consumption_details_expected_main_opt'].extend(detail_plant_consump_exp)
        self.results['hourly_storage_levels_details_expected_main_opt'].extend(detail_hrly_storage_exp)
        self.results['hydrogen_produced_periodical_expected_main_opt'].append({
            'period': period_counter,
            'start_date': time_index_for_impl_period[0].date(),
            'end_date': time_index_for_impl_period[-1].date(),
            'total_expected_hydrogen_kg': expected_h2_total
        })

    def _store_actual_details_and_update_state(self, model_to_use, model_type, period_counter, time_index_for_impl_period, period_end_dt_date):
        act_h2_total=0.0
        plant_ops_act, hrly_storage_act, new_storage, new_plant_ops = [],[],{},{}
        hours_in_implementation = len(time_index_for_impl_period)

        if model_to_use is None:
            self._store_failed_operation_actuals_and_state(period_counter, time_index_for_impl_period[0], hours_in_implementation, period_end_dt_date, "Model None"); return

        total_unused_energy_this_period = sum(pyo.value(model_to_use.Unused_Cleared_Energy_t[t]) for t in model_to_use.H)
        self.total_simulation_unused_energy_mwh += total_unused_energy_this_period
        
        # --- FIX: ADD START DATE AND RENAME COLUMN FOR UNUSED ENERGY ---
        self.results['redispatch_unused_energy_periodic_details'].append({
            'period': period_counter, 
            'start_date': time_index_for_impl_period[0].date(),
            'total_period_unused_mwh': total_unused_energy_this_period
        })
        # --- END FIX ---

        for h_idx in range(hours_in_implementation):
            ts = time_index_for_impl_period[h_idx]
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
        self.results['hydrogen_produced_periodical_actual'].append({'period':period_counter,'start_date':time_index_for_impl_period[0].date(),'end_date':period_end_dt_date,'total_actual_hydrogen_kg':act_h2_total})
        
        if new_storage: self.current_storage_levels=new_storage
        if new_plant_ops: self.current_plant_operating_levels=new_plant_ops
        self.results['storage_levels_period_end'].append({'period':period_counter,'date':period_end_dt_date.strftime('%Y-%m-%d'),'type':model_type,**self.current_storage_levels})

    def _store_failed_operation_actuals_and_state(self, period_counter, period_start_dt, hours_in_implementation, period_end_dt_date, reason="Op Failed"):
        print(f"  Storing ZERO actuals for period {period_counter} due to: {reason}. State carries over.")
        time_idx_impl = pd.date_range(start=period_start_dt, periods=hours_in_implementation, freq='H')
        self.results['hydrogen_produced_periodical_actual'].append({'period':period_counter,'start_date':period_start_dt.date(),'end_date':period_end_dt_date,'total_actual_hydrogen_kg':0.0})
        for ts in time_idx_impl:
            self.results['hourly_site_consumption_actual'].append({'Timestamp': ts, 'Total_Site_Consumption_MWh': self.baseload_mw})
            for p_name in FLEXIBLE_PLANTS: self.results['plant_operations_details_actual_redispatch'].append({'Period':period_counter,'Timestamp':ts,'Plant_Name':p_name,'Actual_Operation_Level':0.0,'Actual_Absolute_Electricity_MWh':0.0})
        
        self.results['storage_levels_period_end'].append({'period':period_counter,'date':period_end_dt_date.strftime('%Y-%m-%d'),'comment':f'{reason}. State carried over.',**self.current_storage_levels})

    def _store_zero_results(self, period_counter, current_date_obj, reason="Optimization failed"):
        implementation_end_date = current_date_obj + timedelta(days=self.implementation_period - 1)
        self.results['costs'].append({'period':period_counter, 'start_date': current_date_obj, 'end_date': implementation_end_date, 'total_accepted_mw':0.0, 'total_cost':0.0})
        self.results['redispatch_summary'].append({'period':period_counter,'success':False,'reason':reason})
        self.results['var_values_main_opt'].append({'period': period_counter, 'VaR_EUR_NetCost': np.nan})
        self.results['cvar_values_main_opt'].append({'period': period_counter, 'CVaR_EUR_NetCost': np.nan})
        self._store_failed_operation_actuals_and_state(period_counter, datetime.combine(current_date_obj, datetime.min.time()), self.implementation_period * 24, implementation_end_date, reason)

    def _calculate_and_store_economic_indicators(self):
        print("\nCalculating final economic and operational indicators for Electrolyzer system...")
        accepted_df = pd.concat(self.results['accepted'], ignore_index=True) if self.results['accepted'] else pd.DataFrame()
        h2_actual_df = pd.DataFrame(self.results['hydrogen_produced_periodical_actual'])
        var_df = pd.DataFrame(self.results['var_values_main_opt'])
        cvar_df = pd.DataFrame(self.results['cvar_values_main_opt'])
        
        indicators = {}
        
        # TOU Calculations
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

        if not accepted_df.empty:
            # Core Economic Calculations
            total_mwh_consumed = accepted_df['Accepted MW'].sum()
            total_market_cost = accepted_df['Cost'].sum()
            total_electricity_cost = total_market_cost + final_tou_cost
            
            indicators['Total MWh Consumed'] = (total_mwh_consumed, 'MWh')
            indicators['Total Electricity Cost (Market + TOU)'] = (total_electricity_cost, 'EUR')
            
            avg_elec_price_paid = (total_electricity_cost / total_mwh_consumed if total_mwh_consumed > 1e-6 else 0)
            indicators['Average Electricity Price Paid (incl. TOU)'] = (avg_elec_price_paid, 'EUR/MWh')

            
            accepted_df['Hourly Price Paid'] = accepted_df['Cost'] / accepted_df['Accepted MW']
            std_dev_hourly_price = accepted_df['Hourly Price Paid'].std()
            indicators['Std Dev of Hourly Price Paid (Risk)'] = (std_dev_hourly_price, 'EUR/MWh')
            
          
            avg_market_price = accepted_df['MCP'].mean()
            indicators['Average Market Price (Benchmark)'] = (avg_market_price, 'EUR/MWh')

            if not h2_actual_df.empty:
                total_h2 = h2_actual_df['total_actual_hydrogen_kg'].sum()
                indicators['Total Hydrogen Produced (Compressed)'] = (total_h2, 'kg')
                
                elec_consumption_per_kg = (total_mwh_consumed * 1000 / total_h2 if total_h2 > 1e-6 else 0)
                indicators['Electricity Consumption per kg H2'] = (elec_consumption_per_kg, 'kWh/kg')
                
                elec_cost_per_kg = (total_electricity_cost / total_h2 if total_h2 > 1e-6 else 0)
                indicators['Electricity Cost per kg H2'] = (elec_cost_per_kg, 'EUR/kg')
                
                h2_price_config = self.config.get('hydrogen_price', {})
                h2_price_per_kg = float(h2_price_config.get('value_eur_per_kg', 0.0))
                total_revenue_from_h2 = total_h2 * h2_price_per_kg
                net_profit = total_revenue_from_h2 - total_electricity_cost
                profit_margin_percentage = (net_profit / total_revenue_from_h2 * 100) if total_revenue_from_h2 > 1e-6 else 0.0
                
                indicators['Total Revenue from Hydrogen'] = (total_revenue_from_h2, 'EUR')
                indicators['Net Profit'] = (net_profit, 'EUR')
                indicators['Profit Margin'] = (profit_margin_percentage, '%')

        
        if not var_df.empty:
            avg_var = var_df['VaR_EUR_NetCost'].mean()
            indicators['Average VaR (NetCost)'] = (avg_var, 'EUR')
            
        if not cvar_df.empty:
            avg_cvar = cvar_df['CVaR_EUR_NetCost'].mean()
            indicators['Average CVaR (NetCost)'] = (avg_cvar, 'EUR')

        analysis_data = [{'Indicator': key, 'Value': val, 'Unit': unit} for key, (val, unit) in indicators.items()]
        self.results['economic_analysis'] = pd.DataFrame(analysis_data)
        print("  Economic and operational indicators calculated successfully.")


    def _save_final_results(self):
        output_prefix = os.path.basename(self.output_dir.rstrip(os.sep))
        self._calculate_and_store_economic_indicators()
        
        if self.synthetic_config.get('enabled', False):
            print("\n--- SAVING SYNTHETIC ACTUAL PRICES (Electrolyzer) ---")
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

        print(f"\n\n{'='*30} FINAL SIMULATION SUMMARY (V5.2 - Full System with Expected Outputs) {'='*30}")
        if not self.results['economic_analysis'].empty:
            print(self.results['economic_analysis'].to_string(index=False))
        for key, data_item in self.results.items():
            if key == 'economic_analysis':
                file_path = os.path.join(self.output_dir, f"{output_prefix}_economic_indicators.xlsx")
            else:
                file_path = os.path.join(self.output_dir, f"{output_prefix}_{key}.xlsx")

            df_to_save = None
            if isinstance(data_item, pd.DataFrame): df_to_save = data_item
            elif isinstance(data_item, list) and data_item:
                if all(isinstance(item, pd.DataFrame) for item in data_item): df_to_save = pd.concat(data_item, ignore_index=True)
                elif all(isinstance(item, dict) for item in data_item): df_to_save = pd.DataFrame(data_item)
            
            if df_to_save is not None and not df_to_save.empty:
                try:
                    df_to_save.to_excel(file_path, index=False, engine='openpyxl')
                    print(f"Saved {key} to {file_path}")
                except Exception as e_save: print(f"Error saving {key}: {e_save}")
        print("\n" + "="*28 + " END OF FINAL SIMULATION SUMMARY " + "="*27 + "\n")
