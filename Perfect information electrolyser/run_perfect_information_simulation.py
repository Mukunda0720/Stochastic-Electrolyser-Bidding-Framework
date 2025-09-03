
"""
Main Script for Running the Perfect Foresight Benchmark Simulation.

This script serves as the entry point for the perfect information case.
It loads the configuration and can now use either actual market prices
from a file or generate synthetic prices for the entire period. It then
calls the planner to solve for the globally optimal operational schedule
in a single run and saves the results.
"""

import argparse
import os
import pandas as pd
import numpy as np
import yaml
import pyomo.environ as pyo
from datetime import datetime, timedelta


from perfect_information_planner import PerfectInformationPlanner
from electrolyzer_definitions import FLEXIBLE_PLANTS, MATERIALS_IN_STORAGE

from synthetic_prices import synth_prices_ar1



def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run Electrolyzer Perfect Information Benchmark.')
    parser.add_argument(
        '--config',
        default='config_perfect_info.yaml',
        help='Path to the config file for the perfect information strategy'
    )
    return parser.parse_args()

#use synthetic price generator if enabled
def _generate_synthetic_prices(sim_config, synth_config):
    """
    Calls the synthetic price generator for the entire simulation period.
    """
    print("INFO: Synthetic data generation is ENABLED. Generating prices...")
    start_dt = pd.to_datetime(sim_config.get('start_date'))
    end_dt = pd.to_datetime(sim_config.get('end_date'))
    total_days = (end_dt - start_dt).days + 1
    total_hours = total_days * 24

    
    gen_params = synth_config.copy()
    gen_params.pop('enabled', None) 


    actual_prices_series, _ = synth_prices_ar1(hours=total_hours, **gen_params)

    # Create a DataFrame in the format the rest of the script expects
    actual_prices_df = pd.DataFrame({
        'Date': pd.date_range(start=start_dt, periods=total_hours, freq='h'),
        'Price': actual_prices_series.values
    })
    print(f"  SUCCESS: Generated {len(actual_prices_df)} hours of synthetic prices.")
    return actual_prices_df



def process_and_save_results(model, config, actual_prices_df):
    """
    Extracts all relevant data from the solved model and saves it to files.
    """
    print("\nProcessing results from the perfect information model...")
    sim_conf = config.get('simulation', {})
    output_dir = sim_conf.get('output_dir')
    start_date = pd.to_datetime(sim_conf.get('start_date'))
    end_date = pd.to_datetime(sim_conf.get('end_date'))
    total_hours = (end_date - start_date).days * 24 + 24
    time_index = pd.date_range(start=start_date, periods=total_hours, freq='h')

    # --- Extract Data from Model ---
    plant_ops = []
    storage_levels = []
    site_consumption = []

    baseload = config.get('electrolyzer_plant', {}).get('baseload_mw', 0.0)

    for t in model.H:
        ts = time_index[t]
        # Plant operations
        for p in model.PLANTS_FLEX:
            plant_ops.append({
                'Timestamp': ts,
                'Plant_Name': p,
                'Operation_Level': pyo.value(model.P_plant[p, t]),
                'Is_On': pyo.value(model.Plant_Is_On_t[p, t])
            })
        # Storage levels (end of hour)
        for m in model.MATERIALS:
            storage_levels.append({
                'Timestamp': ts,
                'Material': m,
                'Storage_Level': pyo.value(model.S_material[m, t + 1])
            })
        # Site consumption
        site_consumption.append({
            'Timestamp': ts,
            'Flexible_Consumption_MWh': pyo.value(model.Total_Electricity_MWh[t]),
            'Baseload_MWh': baseload,
            'Total_Site_Consumption_MWh': pyo.value(model.Total_Electricity_MWh[t]) + baseload
        })

    plant_ops_df = pd.DataFrame(plant_ops)
    storage_levels_df = pd.DataFrame(storage_levels)
    site_consumption_df = pd.DataFrame(site_consumption)

    # --- Calculate Final Economic Summary ---
    total_mwh = site_consumption_df['Total_Site_Consumption_MWh'].sum()
    total_revenue = pyo.value(sum(model.P_plant['Compressor', t] for t in model.H) * model.P_hydrogen)
    
    market_cost = pyo.value(sum(model.Total_Electricity_MWh[t] * model.Actual_Price[t] for t in model.H))
    
    tou_cost = 0
    peak_chargeable_mw = 0.0
    
    tou_config = config.get('time_of_use_tariff',{})
    if tou_config.get('enabled'):
        scaling_factor = pyo.value(model.TotalDays) / 30.4
        tou_rate = tou_config.get('rate_eur_per_kw_month', 0.0)
        
        # Extract the weighted peak consumption value from the model
        peak_chargeable_mw = pyo.value(model.E_peak_chargeable)
        
        # Calculate the TOU cost based on this peak value
        tou_cost = peak_chargeable_mw * 1000 * tou_rate * scaling_factor

    total_electricity_cost = market_cost + tou_cost
    net_profit = total_revenue - total_electricity_cost
    total_final_product = pyo.value(sum(model.P_plant['Compressor', t] for t in model.H))

    # --- NEW: Calculate profit margin ---
    profit_margin_percentage = (net_profit / total_revenue * 100) if total_revenue > 1e-6 else 0.0
    
    summary_data = {
        'Indicator': [
            'Total Net Profit', 'Total Revenue from Hydrogen', 'Total Electricity Cost (Market + TOU)',
            '  - Market Cost', '  - TOU Peak Charge', 'Total MWh Consumed',
            'Total Final Product (Compressed H2)', 'Average Electricity Price Paid',
            'Cost per kg of Final Product', 'Revenue per kg of Final Product'
        ],
        'Value': [
            net_profit, total_revenue, total_electricity_cost, market_cost, tou_cost,
            total_mwh, total_final_product,
            total_electricity_cost / total_mwh if total_mwh > 1e-6 else 0,
            total_electricity_cost / total_final_product if total_final_product > 1e-6 else 0,
            pyo.value(model.P_hydrogen)
        ],
        'Unit': [
            'EUR', 'EUR', 'EUR', 'EUR', 'EUR', 'MWh', 'kg', 'EUR/MWh', 'EUR/kg', 'EUR/kg'
        ]
    }

    
    try:
        insert_index = summary_data['Indicator'].index('Total Net Profit') + 1
        summary_data['Indicator'].insert(insert_index, 'Profit Margin')
        summary_data['Value'].insert(insert_index, profit_margin_percentage)
        summary_data['Unit'].insert(insert_index, '%')
    except ValueError:
        pass # Fallback

    
    if tou_config.get('enabled'):
        try:
            insert_index = summary_data['Indicator'].index('  - TOU Peak Charge') + 1
            summary_data['Indicator'].insert(insert_index, '  - TOU Weighted Peak')
            summary_data['Value'].insert(insert_index, peak_chargeable_mw)
            summary_data['Unit'].insert(insert_index, 'MW')
        except ValueError:
            print("Warning: Could not find '  - TOU Peak Charge' to insert weighted peak metric.")

    summary_df = pd.DataFrame(summary_data)

    # --- Save to Files ---
    output_prefix = os.path.basename(output_dir.rstrip(os.sep))
    plant_ops_df.to_excel(os.path.join(output_dir, f"{output_prefix}_hourly_plant_operations.xlsx"), index=False)
    storage_levels_df.to_excel(os.path.join(output_dir, f"{output_prefix}_hourly_storage_levels.xlsx"), index=False)
    site_consumption_df.to_excel(os.path.join(output_dir, f"{output_prefix}_hourly_site_consumption.xlsx"), index=False)
    summary_df.to_excel(os.path.join(output_dir, f"{output_prefix}_economic_summary.xlsx"), index=False)

    
    if config.get('synthetic_data', {}).get('enabled', False):
        print("\nSaving synthetic actual prices used in simulation...")
        synth_prices_to_save = actual_prices_df.copy()
        synth_prices_to_save.rename(columns={'Price': 'Synthetic_Actual_MCP'}, inplace=True)
        fpath_actuals = os.path.join(output_dir, f"{output_prefix}_synthetic_actual_prices.xlsx")
        synth_prices_to_save.to_excel(fpath_actuals, index=False)
        print(f"Saved synthetic prices to {fpath_actuals}")
   


    print("\n" + "="*30 + " FINAL SIMULATION SUMMARY (Perfect Info) " + "="*30)
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to directory: {output_dir}")
    print("="*80)


def main():
    """Main function to run the simulation."""
    args = parse_arguments()

    if not os.path.exists(args.config):
        print(f"FATAL ERROR: Configuration file '{args.config}' not found.")
        return 1

    try:
        start_time = datetime.now()
        print(f"Starting Perfect Information simulation at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Using configuration: {args.config}")

        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        # Conditional data loading ---
        sim_conf = config.get('simulation', {})
        synth_conf = config.get('synthetic_data', {})
        
        actual_prices_df = None
        if synth_conf.get('enabled', False):
            # Generate prices synthetically
            actual_prices_df = _generate_synthetic_prices(sim_conf, synth_conf)
        else:
            # Original behavior: load from file
            print(f"INFO: Synthetic data is DISABLED. Loading from {sim_conf.get('market_input_file')}")
            market_file = sim_conf.get('market_input_file')
            start_date = pd.to_datetime(sim_conf.get('start_date'))
            end_date = pd.to_datetime(sim_conf.get('end_date'))
            total_days = (end_date - start_date).days + 1

            market_data = pd.read_excel(market_file)
            market_data['Date'] = pd.to_datetime(market_data['Date'])
            
            # Filter prices for the exact simulation period
            actual_prices_df = market_data[
                (market_data['Date'] >= start_date) &
                (market_data['Date'] < start_date + timedelta(days=total_days))
            ].sort_values('Date')
        

        # The rest of the logic remains the same, using the `actual_prices_df`
        # which is now populated either from a file or the generator.
        start_date = pd.to_datetime(sim_conf.get('start_date'))
        end_date = pd.to_datetime(sim_conf.get('end_date'))
        total_days = (end_date - start_date).days + 1
        total_hours = total_days * 24
        
        if len(actual_prices_df) != total_hours:
            raise ValueError(f"Price data has {len(actual_prices_df)} hours, but simulation period requires {total_hours}.")

        actual_prices_dict = dict(zip(range(total_hours), actual_prices_df['Price']))

        # Instantiate and run the planner
        planner = PerfectInformationPlanner(config)
        model, results = planner.optimize_for_entire_period(
            actual_prices=actual_prices_dict,
            start_date=start_date,
            total_days=total_days
        )

        # Process and save results
        if results.solver.termination_condition in [pyo.TerminationCondition.optimal, pyo.TerminationCondition.locallyOptimal]:
            process_and_save_results(model, config, actual_prices_df)
        else:
            print("\nCRITICAL: Solver did not find an optimal solution.")
            print(f"Solver status: {results.solver.status}, Termination condition: {results.solver.termination_condition}")

        end_time = datetime.now()
        elapsed = end_time - start_time
        print(f"\nPerfect Information simulation completed in {elapsed.total_seconds():.1f} seconds.")
        return 0

    except Exception as e:
        print(f"\nERROR: An unexpected error occurred during the simulation: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)