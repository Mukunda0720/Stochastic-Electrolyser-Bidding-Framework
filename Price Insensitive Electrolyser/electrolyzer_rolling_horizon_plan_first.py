
"""
Rolling Horizon Manager for the Deterministic "Plan First, Buy Later" Strategy.
This version orchestrates the simulation for a complex system including an
electrolyzer, hydrogen storage, and a compressor. It can now use either
file-based market prices or generated synthetic prices.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import os
import yaml
import traceback
import time
import pyomo.environ as pyo


from deterministic_electrolyzer_planner import DeterministicElectrolyzerPlanner 
from simple_market_clearing import SimpleMarketClearing 

from electrolyzer_definitions import (
    PLANT_DEFINITIONS,
    FLEXIBLE_PLANTS,
    ELECTRICITY_INPUT_MAP,
    STORAGE_DEFINITIONS,
    MATERIALS_IN_STORAGE
)
# --- MODIFICATION: Import the new synthetic price generator ---
from synthetic_prices import synth_prices_ar1 


class ElectrolyzerRollingHorizonManagerPlanFirst:
    """
    Orchestrates the "Plan First, Buy Later" simulation for the complex system.
    """
    def __init__(self, config_file='config_plan_first.yaml'): 
        print(f"Initializing ElectrolyzerRollingHorizonManagerPlanFirst (V2.1) from config: {config_file}")
        try:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            print(f"FATAL ERROR: Could not load or parse config file '{config_file}': {e}"); raise

        # Load parameters from config
        sim_params = self.config.get('simulation', {}) 
        self.forecast_params = self.config.get('forecast', {}) 
        self.electrolyzer_plant_config = self.config.get('electrolyzer_plant', {}) 

        self.synthetic_config = self.config.get('synthetic_data', {'enabled': False}) 

        self.start_date = pd.to_datetime(sim_params.get('start_date')).tz_localize(None) 
        self.end_date = pd.to_datetime(sim_params.get('end_date')).tz_localize(None) 
        self.planning_horizon = int(sim_params.get('planning_horizon', 5)) 
        self.implementation_period = int(sim_params.get('implementation_period', 1)) 
        self.market_input_file = sim_params.get('market_input_file') 
        self.output_dir = sim_params.get('output_dir') 
        os.makedirs(self.output_dir, exist_ok=True) 

        self.random_seed = int(sim_params.get('random_seed', 42)) 
        np.random.seed(self.random_seed) 

        self.market_clearing = SimpleMarketClearing({'output_dir': self.output_dir}) 
        self.planner_instance = DeterministicElectrolyzerPlanner(self.config) 
        self.planner_instance.output_dir = self.output_dir 

        # --- MODIFIED: Initialize state variables for plants AND storage ---
        self.current_plant_operating_levels = { 
            p: self.planner_instance._get_plant_abs_limit(p, 'Min_Op_Abs', 0.0)
            for p in FLEXIBLE_PLANTS 
        }
        self.current_storage_levels = { 
            mat: STORAGE_DEFINITIONS[mat]['Initial_Level'] for mat in MATERIALS_IN_STORAGE 
        }

        self._reset_results() 

        self.target_hydrogen_per_day_benchmark = self.electrolyzer_plant_config.get('target_hydrogen_per_day') 
        self.total_simulation_days = (self.end_date.date() - self.start_date.date()).days + 1 
        self.total_simulation_benchmark_h2_overall = self.target_hydrogen_per_day_benchmark * self.total_simulation_days 

        self.general_market_price_cap = self.forecast_params.get('prophet', {}).get('cap', 4000.0) 
        self.baseload_mw = self.electrolyzer_plant_config.get('baseload_mw', 0.0) 

        print(f"Output directory set to: {self.output_dir}")
        print(f"Strategy: V2.1 'Plan First, Buy Later' (Complex System)")
        self._load_market_data() 

    # --- MODIFICATION: Add the new helper method to generate prices ---
    def _generate_synthetic_prices(self, start_dt, horizon_days): 
        """
        Calls the external synth_prices_ar1 generator with parameters from the config file.
        """
        cfg = self.synthetic_config.copy() 
        cfg.pop('enabled', None) 

        total_hours = horizon_days * 24 

        actual_s, forecast_s = synth_prices_ar1(hours=total_hours, **cfg) 

        # Align the index with the simulation timeline
        actual_s.index = pd.date_range(start=start_dt, periods=total_hours, freq='H') 
        forecast_s.index = pd.date_range(start=start_dt, periods=total_hours, freq='H') 

        # Create DataFrames in the expected format
        forecast_df = pd.DataFrame({'Date': forecast_s.index, 'Forecast_Price': forecast_s.values}) 
        # The 'Price' column name is critical, as it's what simple_market_clearing expects
        actuals_df = pd.DataFrame({'Date': actual_s.index, 'Price': actual_s.values}) 

        print(f"  SYNTHETIC DATA: Generated {total_hours}h of prices using synth_prices_ar1.")
        return forecast_df, actuals_df

    def _reset_results(self): #
        """Resets all results containers for a new simulation run."""
        self.results = { #
            'bids': [], 'accepted': [], 'costs': [],
            'plant_operations_details_actual': [],
            'final_product_produced_periodical_actual': [], 
            'hourly_energy_plan': [],
            'hourly_site_consumption_actual': [],
            'hourly_storage_levels_actual': [], 
            'economic_analysis': []
        }
        self.total_implemented_final_product_actual_so_far = 0.0 

    def _load_market_data(self): 
        """
        Loads and prepares the market price data for clearing.
        MODIFIED: Now supports generating synthetic data instead of reading from a file.
        """
        # Use synthetic data generator if enabled in config ---
        if self.synthetic_config.get('enabled', False): #
            print("INFO: Synthetic data generation is ENABLED. Bypassing market_input_file.")
            # Calculate total simulation days plus the planning horizon for a buffer
            total_sim_days = (self.end_date - self.start_date).days + self.planning_horizon + 1 

            
            _, full_actuals_df = self._generate_synthetic_prices(self.start_date, total_sim_days) 

            self.market_data = full_actuals_df #
            print(f"Loaded {len(self.market_data)} rows of synthetic 'actual' prices into memory.")
            return


        try:
            print(f"INFO: Synthetic data is DISABLED. Loading market data from {self.market_input_file}")
            market_data_df = pd.read_excel(self.market_input_file) 
            market_data_df['Date'] = pd.to_datetime(market_data_df['Date']) 
            if market_data_df['Date'].dt.tz is not None: 
                market_data_df['Date'] = market_data_df['Date'].dt.tz_localize(None) 
            self.market_data = market_data_df #
            print(f"Loaded market data for clearing from {self.market_input_file}")
        except Exception as e:
            print(f"ERROR loading market data: {e}"); raise RuntimeError("Failed to load market data") from e

    def run_simulation(self): 
        """The main loop for the rolling horizon simulation."""
        print(f"\n===== Starting 'Plan First' Simulation (V2.1) =====")
        current_dt = self.start_date 
        period_counter = 1 

        while current_dt.date() <= self.end_date.date(): 
            period_start_time = time.time() 
            print(f"\n{'='*80}\nStarting planning period {period_counter}: {current_dt.strftime('%Y-%m-%d')}\n{'='*80}")
            print(f"  Current Storage: {{ {', '.join([f'{k}: {v:.2f}' for k,v in self.current_storage_levels.items()])} }}")
            print(f"  Current Plant Ops: {{ {', '.join([f'{k}: {v:.2f}' for k,v in self.current_plant_operating_levels.items()])} }}")

            days_remaining_in_sim = (self.end_date.date() - current_dt.date()).days + 1 
            days_in_horizon_window = min(self.planning_horizon, days_remaining_in_sim) 

            if days_in_horizon_window < self.implementation_period: 
                print("Stopping simulation. Not enough future days for the implementation period.")
                break

            remaining_production = self.total_simulation_benchmark_h2_overall - self.total_implemented_final_product_actual_so_far 
            days_left_in_simulation = max(1, (self.end_date.date() - current_dt.date()).days + 1) 
            dynamic_daily_target = remaining_production / days_left_in_simulation 
            production_target_for_this_window = dynamic_daily_target * days_in_horizon_window 

            print(f"  Dynamic Target for this Window = {production_target_for_this_window:,.0f} kg")

            print(f"\nAttempting DETERMINISTIC planning...")
            
            planned_model, plan_results = self.planner_instance.plan_schedule_for_target( 
                production_target_for_window=production_target_for_this_window,
                horizon_days=days_in_horizon_window,
                initial_plant_operating_levels=self.current_plant_operating_levels,
                initial_storage_levels=self.current_storage_levels, 
                maintenance_schedule=None
            )

            if plan_results is None or plan_results.solver.termination_condition != pyo.TerminationCondition.optimal: 
                print(f"CRITICAL: Deterministic planning failed for period {period_counter}. Stopping simulation.")
                break
            print("Deterministic Planning SUCCEEDED.")

            time_index_for_impl_period = pd.date_range(start=current_dt, periods=self.implementation_period*24, freq='h') 

            hourly_energy_plan = [{'Period': period_counter, 'Timestamp': time_index_for_impl_period[h_idx], 'Planned_Energy_MWh': pyo.value(planned_model.Energy_consumption_t[h_idx])} for h_idx in range(self.implementation_period * 24)] #
            self.results['hourly_energy_plan'].extend(hourly_energy_plan) 

            bid_records = [{"Date": item['Timestamp'], "Bid Price": self.general_market_price_cap, "Bid Quantity (MW)": item['Planned_Energy_MWh'] + self.baseload_mw} for item in hourly_energy_plan if item['Planned_Energy_MWh'] + self.baseload_mw > 1e-3] #
            bids_df = pd.DataFrame(bid_records) 
            self.results['bids'].append(bids_df) 

            accepted_bids_df = self.market_clearing.clear_market(bids_df, self.market_data) 
            summary_mc = self.market_clearing.calculate_summary(accepted_bids_df) 
            self.results['accepted'].append(accepted_bids_df) #
            self.results['costs'].append({'period': period_counter, 'total_cost_cleared': summary_mc.get('total_cost', 0.0)}) 

            self._update_state_and_store_results(planned_model, period_counter, time_index_for_impl_period) 
            print(f"\n--- Period {period_counter} processing finished in {time.time() - period_start_time:.2f} seconds ---")
            current_dt += timedelta(days=self.implementation_period) #
            period_counter += 1 

        self._save_final_results() 
        return self.results 

    def _update_state_and_store_results(self, planned_model, period_counter, time_index_for_impl_period): 
        """Stores results from the implemented plan and updates the system state."""
        hours_in_implementation = self.implementation_period * 24 
        
        # --- MODIFIED: Track final product and update storage state ---
        final_product_total = 0.0
        detail_plant_ops_act = []
        detail_storage_act = []

        for h_idx in range(hours_in_implementation):
            ts = time_index_for_impl_period[h_idx]

            # The final product is the output of the compressor
            final_product_total += pyo.value(planned_model.P_plant['Compressor', h_idx])

            total_site_consumption_mwh = pyo.value(planned_model.Energy_consumption_t[h_idx]) + self.baseload_mw
            self.results['hourly_site_consumption_actual'].append({'Period': period_counter, 'Timestamp': ts, 'Total_Site_Consumption_MWh': total_site_consumption_mwh})

            for p_name in FLEXIBLE_PLANTS:
                op_level = pyo.value(planned_model.P_plant[p_name, h_idx])
                detail_plant_ops_act.append({'Period': period_counter, 'Timestamp': ts, 'Plant_Name': p_name, 'Actual_Operation_Level': op_level})
                if h_idx == hours_in_implementation - 1:
                    self.current_plant_operating_levels[p_name] = op_level

            # Log hourly storage levels and update state for the next period
            for mat in MATERIALS_IN_STORAGE:
                storage_level = pyo.value(planned_model.S_material[mat, h_idx + 1])
                detail_storage_act.append({'Period': period_counter, 'Timestamp': ts, 'Material': mat, 'Actual_Storage_Level': storage_level})
                if h_idx == hours_in_implementation - 1:
                    self.current_storage_levels[mat] = storage_level

        self.results['plant_operations_details_actual'].extend(detail_plant_ops_act)
        self.results['hourly_storage_levels_actual'].extend(detail_storage_act)
        self.results['final_product_produced_periodical_actual'].append({'period': period_counter, 'total_actual_final_product_kg': final_product_total})
        self.total_implemented_final_product_actual_so_far += final_product_total

    def _get_tou_weighting_factors_for_timestamp(self, timestamp):
        """Provides weighting factors for Time-of-Use tariff calculations."""
        # This function is unchanged but is now used for economic analysis.
        weekday_weights = {1:[0.7,0.7,0.7,0.7,0.7,0.7,0.8,0.9,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.9,0.8],
                           2:[0.7,0.7,0.7,0.7,0.7,0.7,0.8,0.9,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.9,0.8],
                           3:[0.7,0.7,0.7,0.7,0.7,0.7,0.8,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,1.0,1.0,1.0,1.0,0.9,0.8,0.8],
                           4:[0.7,0.7,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.7,0.6,0.6,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.8,0.8,0.8,0.8,0.8],
                           5:[0.7,0.7,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.7,0.6,0.6,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.8,0.8,0.8,0.8,0.8],
                           6:[0.7,0.7,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.7,0.6,0.6,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.8,0.8,0.8,0.8,0.8],
                           7:[0.7,0.7,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.7,0.6,0.6,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.8,0.8,0.8,0.8,0.8],
                           8:[0.7,0.7,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.7,0.6,0.6,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.8,0.8,0.8,0.8,0.8],
                           9:[0.7,0.7,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.7,0.6,0.6,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.8,0.8,0.8,0.8,0.8], 
                           10:[0.7,0.7,0.7,0.7,0.7,0.7,0.8,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,1.0,1.0,1.0,1.0,0.9,0.8], 
                           11:[0.7,0.7,0.7,0.7,0.7,0.7,0.8,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,1.0,1.0,1.0,1.0,0.9,0.8], 
                           12:[0.7,0.7,0.7,0.7,0.7,0.7,0.8,0.9,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.9,0.8]} 
        weekend_weights = [0.7,0.7,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.7,0.6,0.6,0.6,0.6,0.6,0.6,0.7,0.8,0.8,0.8,0.8,0.8,0.8,0.8] 
        try:
            import holidays
            nl_holidays = holidays.country_holidays('NL', years=timestamp.year) 
        except ImportError: nl_holidays = {}
        is_weekend = timestamp.dayofweek >= 5 
        is_holiday = timestamp.date() in nl_holidays 
        if is_weekend or is_holiday: return weekend_weights[timestamp.hour] 
        else: return weekday_weights[timestamp.month][timestamp.hour] 

    def _calculate_and_store_economic_indicators(self): 
        """Calculates final economic and operational metrics for the whole simulation."""
        print("\nCalculating final economic and operational indicators...")
        accepted_df = pd.concat(self.results['accepted'], ignore_index=True) if self.results['accepted'] else pd.DataFrame() 
        final_product_df = pd.DataFrame(self.results['final_product_produced_periodical_actual']) 
        indicators = {} 
        final_tou_cost = 0.0 
        tou_config = self.config.get('time_of_use_tariff', {}) 

        if tou_config.get('enabled', False) and self.results['hourly_site_consumption_actual']: 
            consumption_df = pd.DataFrame(self.results['hourly_site_consumption_actual'])
            if not consumption_df.empty: 
                consumption_df['Timestamp'] = pd.to_datetime(consumption_df['Timestamp'])
                consumption_df['weight'] = consumption_df['Timestamp'].apply(self._get_tou_weighting_factors_for_timestamp)
                consumption_df['weighted_consumption_mw'] = consumption_df['Total_Site_Consumption_MWh'] * consumption_df['weight']
                simulation_peak_mw = consumption_df['weighted_consumption_mw'].max()
                tou_rate = tou_config.get('rate_eur_per_kw_month', 0.0)
                final_tou_cost = simulation_peak_mw * 1000 * tou_rate
                indicators['TOU - Simulation Weighted Peak'] = (simulation_peak_mw, 'MW')
                indicators['TOU - Final Peak Charge'] = (final_tou_cost, 'EUR')

        if not accepted_df.empty and not final_product_df.empty: 
            total_mwh_consumed = accepted_df['Accepted MW'].sum() 
            total_market_cost = accepted_df['Cost'].sum() 
            total_electricity_cost = total_market_cost + final_tou_cost
            total_final_product = final_product_df['total_actual_final_product_kg'].sum()
            h2_price_per_kg = self.config.get('hydrogen_price', {}).get('value_eur_per_kg', 8.0)
            total_revenue = total_final_product * h2_price_per_kg
            net_profit = total_revenue - total_electricity_cost
            profit_margin_percentage = (net_profit / total_revenue * 100) if total_revenue > 1e-6 else 0.0

            indicators.update({
                'Total MWh Consumed': (total_mwh_consumed, 'MWh'),
                'Total Electricity Cost (Market + TOU)': (total_electricity_cost, 'EUR'),
                'Average Electricity Price Paid': (total_electricity_cost / total_mwh_consumed if total_mwh_consumed > 1e-6 else 0, 'EUR/MWh'),
                'Total Final Product (Compressed H2)': (total_final_product, 'kg'),
                'Electricity Consumption per kg H2': (total_mwh_consumed * 1000 / total_final_product if total_final_product > 1e-6 else 0, 'kWh/kg'),
                'Total Revenue from Final Product': (total_revenue, 'EUR'),
                'Net Profit': (net_profit, 'EUR'),
                'Profit Margin': (profit_margin_percentage, '%'),
                'Cost per kg of Final Product': (total_electricity_cost / total_final_product if total_final_product > 1e-6 else 0, 'EUR/kg'),
            })

            if 'MCP' in accepted_df.columns:
                hourly_prices = accepted_df.groupby('Date')['MCP'].first()
                indicators['Std Dev of Hourly Price Paid (Risk)'] = (hourly_prices.std(), 'EUR/MWh')
                indicators['Average Market Price (Benchmark)'] = (hourly_prices.mean(), 'EUR/MWh')


        self.results['economic_analysis'] = pd.DataFrame([{'Indicator': key, 'Value': val, 'Unit': unit} for key, (val, unit) in indicators.items()])
        print("  Economic and operational indicators calculated successfully.")

    def _save_final_results(self): 
        """Saves all collected results to Excel files."""
        self._calculate_and_store_economic_indicators()
        output_prefix = os.path.basename(self.output_dir.rstrip(os.sep))
        print(f"\n\n{'='*30} FINAL SIMULATION SUMMARY (Plan First - Complex) {'='*30}")
        print(f"  Total ACTUAL Implemented Final Product: {self.total_implemented_final_product_actual_so_far:,.2f} kg")
        target_total = self.total_simulation_benchmark_h2_overall
        print(f"  Overall Simulation Target: {target_total:,.2f} kg")
        pct_met = (self.total_implemented_final_product_actual_so_far / target_total * 100) if target_total > 0 else 0
        print(f"  Percentage of Target Met: {pct_met:.2f}%")

        # --- MODIFICATION: Save the synthetic actual prices if they were used ---
        if self.synthetic_config.get('enabled', False):
            print("\n--- SAVING SYNTHETIC ACTUAL PRICES (Plan-First) ---")
            try:
                # Filter the market data to the actual simulation range
                end_of_sim_dt = self.end_date + timedelta(days=1) - timedelta(seconds=1)
                sim_actuals_df = self.market_data[
                    (self.market_data['Date'] >= self.start_date) &
                    (self.market_data['Date'] <= end_of_sim_dt)
                ].copy()

                sim_actuals_df.rename(columns={'Price': 'Synthetic_Actual_MCP'}, inplace=True)

                fpath_actuals = os.path.join(self.output_dir, f"{output_prefix}_synthetic_actual_prices.xlsx")
                sim_actuals_df.to_excel(fpath_actuals, index=False)
                print(f"Saved synthetic actual prices to {fpath_actuals}")

            except Exception as e_save_synth:
                print(f"Error saving synthetic actual prices: {e_save_synth}")
        # --- END MODIFICATION ---

        for key, data_item in self.results.items():
            if data_item is None or (isinstance(data_item, (list, dict, pd.DataFrame)) and not len(data_item)):
                continue
            df_to_save = None
            try:
                if isinstance(data_item, list) and data_item:
                    df_to_save = pd.DataFrame(data_item) if isinstance(data_item[0], dict) else pd.concat(data_item, ignore_index=True)
                elif isinstance(data_item, pd.DataFrame):
                    df_to_save = data_item
            except (IndexError, TypeError): continue

            if df_to_save is not None and not df_to_save.empty:
                try:
                    file_path = os.path.join(self.output_dir, f"{output_prefix}_{key}.xlsx")
                    df_to_save.to_excel(file_path, index=False)
                    print(f"Saved {key} to {file_path}")
                except Exception as e:
                    print(f"Error saving {key} to Excel: {e}")
        print("\n" + "="*28 + " END OF FINAL SIMULATION SUMMARY " + "="*27 + "\n")