# plot_steel_plant_simulation_results.py
# DESCRIPTION: Adapted from the electrolyzer plotting script to visualize steel plant simulation results.
# MODIFIED: Added default paths to run directly in a standard project setup.

"""
Generates plots for visualizing steel plant simulation results including:
- A simplified plant analysis focusing on electricity, market price, and deviation.
- Improved price scenario "fan charts" for clear percentile visualization.
- Cumulative production plot for Hot Rolled Steel with targets and market context.
- Storage levels for materials like Coke, Iron, Pellets, Steel, etc.
- Price forecasts vs. actuals (directly from CSVs).
- Bidding curves for daily market participation.

Usage:
    python plot_steel_plant_simulation_results.py --output_dir /path/to/your/simulation_output_directory --config /path/to/your/config.yaml
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import argparse
import sys
import yaml
from datetime import timedelta
import traceback
import numpy as np

# --- ADAPTATION: Import definitions for the STEEL PLANT system ---
try:
    from steel_plant_definitions import PLANT_DEFINITIONS, ABSOLUTE_PLANT_LIMITS, STORAGE_DEFINITIONS
    from ml_forecaster import ElectricityPriceForecaster # Needed for its _calculate_metrics_dict
    from ml_scenario_generator import ScenarioGenerator # Needed for its plot_scenarios
except ImportError as e:
    print(f"ERROR: Could not import necessary modules: {e}")
    print("Ensure steel_plant_definitions.py, ml_forecaster.py, and ml_scenario_generator.py are accessible.")
    sys.exit(1)

def load_config(config_file_path):
    """Loads and parses the YAML configuration file."""
    if not os.path.exists(config_file_path):
        print(f"ERROR: Configuration file not found: {config_file_path}")
        return None
    try:
        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading or parsing config file {config_file_path}: {e}")
        return None

def get_actual_planning_horizon_for_window(window_start_date, config):
    """
    Determines the actual planning horizon for a given window,
    considering the simulation end date to avoid overshooting.
    """
    sim_config = config.get('simulation', {})
    planning_horizon_days_config = sim_config.get('planning_horizon', 5) # Default for steel plant
    sim_end_date_str = sim_config.get('end_date')

    if not sim_end_date_str:
        return planning_horizon_days_config

    try:
        sim_end_date_config = pd.to_datetime(sim_end_date_str).tz_localize(None)
        window_start_dt = pd.to_datetime(window_start_date).tz_localize(None)
        days_left_in_sim = (sim_end_date_config.date() - window_start_dt.date()).days + 1
        return max(min(planning_horizon_days_config, days_left_in_sim), 1)
    except Exception as e:
        print(f"Error in get_actual_planning_horizon_for_window: {e}.")
        return planning_horizon_days_config

def load_steel_plant_data(output_dir, config):
    """Loads all necessary result files from the steel plant simulation output directory."""
    sim_config = config.get('simulation', {})
    config_output_dir = sim_config.get('output_dir', './steel_plant_results')
    output_dir_basename = os.path.basename(config_output_dir.rstrip('/\\'))

    print(f"DEBUG: Using output file prefix derived from config: '{output_dir_basename}'")

    # --- ADAPTATION: Changed filenames to match steel plant outputs ---
    files_to_try = {
        "consumption_data_expected": f"{output_dir_basename}_plant_energy_consumption_details_expected_main_opt.xlsx",
        "storage_data_hourly_expected": f"{output_dir_basename}_hourly_storage_levels_details_expected_main_opt.xlsx",
        "hot_rolled_steel_prod_file_expected": f"{output_dir_basename}_hot_rolled_steel_implemented_periodical_expected_main_opt.xlsx",
        "consumption_data_actual": f"{output_dir_basename}_plant_operations_details_actual_redispatch.xlsx",
        "storage_data_hourly_actual": f"{output_dir_basename}_hourly_storage_levels_details_actual_redispatch.xlsx",
        "hot_rolled_steel_prod_file_actual": f"{output_dir_basename}_hot_rolled_steel_implemented_periodical_actual_redispatch.xlsx",
        "unused_energy_df": f"{output_dir_basename}_redispatch_unused_energy_periodic_details.xlsx",
        "accepted_bids_df_full": f"{output_dir_basename}_accepted.xlsx",
        "all_bids_made_df": f"{output_dir_basename}_bids.xlsx",
        "forecast_periods_info": f"{output_dir_basename}_forecast_errors.xlsx",
    }

    market_input_file_from_config = config.get('simulation', {}).get('market_input_file', None)
    loaded_data = {}

    for key, filename in files_to_try.items():
        filepath = os.path.join(output_dir, filename)
        df_to_assign = pd.DataFrame()
        if os.path.exists(filepath):
            try:
                df = pd.read_excel(filepath)
                for col in ['Timestamp', 'Date', 'start_date', 'end_date']:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce').dt.tz_localize(None)
                df_to_assign = df
            except Exception as e:
                print(f"Warning: Error loading or processing {key} from {filepath}: {e}")
        else:
            print(f"Warning: File not found for {key}: {filepath}")
        loaded_data[key] = df_to_assign

    market_data_full = pd.DataFrame()
    if market_input_file_from_config and os.path.exists(market_input_file_from_config):
        try:
            target_var = config.get('forecast', {}).get('target_variable', 'Price')
            temp_forecaster = ElectricityPriceForecaster({'target_variable': target_var})
            market_data_full = temp_forecaster.load_data(market_input_file_from_config).rename(columns={target_var: 'MCP', 'Date': 'Timestamp'})
            print(f"Successfully loaded full market data from {market_input_file_from_config}")
        except Exception as e:
            print(f"Error loading full market data: {e}")
    else:
        print("Warning: Full market data file not specified or not found. Cannot plot a complete MCP line.")

    return loaded_data, market_data_full


def plot_forecast_vs_actual_window(period_info, config, market_data_full, main_output_dir, plot_output_dir):
    """Plots the price forecast for a single window against the actual prices."""
    if period_info is None or pd.isna(period_info.get('period')) or pd.isna(period_info.get('start_date')):
        return

    period = int(period_info['period'])
    window_start_date = pd.to_datetime(period_info['start_date']).tz_localize(None)
    actual_horizon_for_window = get_actual_planning_horizon_for_window(window_start_date, config)

    forecast_load_dir = os.path.join(main_output_dir, 'forecast_outputs_per_window')
    forecast_file_name = f"forecast_p{period}_{window_start_date.strftime('%Y%m%d')}_h{actual_horizon_for_window}d.csv"
    forecast_file_path = os.path.join(forecast_load_dir, forecast_file_name)

    if not os.path.exists(forecast_file_path):
        print(f"ERROR: Saved forecast file not found: {forecast_file_path}")
        return

    try:
        forecast_df = pd.read_csv(forecast_file_path)
        forecast_df['Date'] = pd.to_datetime(forecast_df['Date']).dt.tz_localize(None)

        forecaster_config = config.get('forecast', {})
        target_variable = forecaster_config.get('target_variable', 'Price')

        actual_df_window = pd.DataFrame()
        if not market_data_full.empty:
            window_end_date_dt = window_start_date + timedelta(days=actual_horizon_for_window) - timedelta(seconds=1)
            actual_df_window = market_data_full[
                (market_data_full['Timestamp'] >= window_start_date) &
                (market_data_full['Timestamp'] <= window_end_date_dt)
            ].copy()

        fig, ax1 = plt.subplots(figsize=(15, 7))
        ax1.plot(forecast_df['Date'], forecast_df['Forecast_Price'], label='Forecast', color='blue', linewidth=1.5)

        metrics_text = "Forecast Only"
        if not actual_df_window.empty:
            merged_df = pd.merge(forecast_df, actual_df_window[['Timestamp', 'MCP']], left_on='Date', right_on='Timestamp', how='left')

            if not merged_df['MCP'].isnull().all():
                ax1.plot(merged_df['Date'], merged_df['MCP'], label='Actual', color='red', alpha=0.6, linewidth=1.0, linestyle='--')
                valid_comparison_df = merged_df.dropna(subset=['MCP', 'Forecast_Price'])
                if not valid_comparison_df.empty:
                    temp_forecaster = ElectricityPriceForecaster(forecaster_config)
                    metrics_dict = temp_forecaster._calculate_metrics_dict(
                        valid_comparison_df['MCP'].values,
                        valid_comparison_df['Forecast_Price'].values
                    )
                    metrics_text = '\n'.join([f"{k.upper()}: {v:.2f}" for k, v in metrics_dict.items()])

        ax1.set_title(f'Period {period}: Price Forecast vs Actual', fontsize=14)
        ax1.set_xlabel('Date'); ax1.set_ylabel('Price (€/MWh)')
        ax1.legend(loc='upper left'); ax1.grid(True, alpha=0.4)

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, fontsize=9, verticalalignment='top', bbox=props)
        fig.autofmt_xdate(); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(plot_output_dir, f"period_{period}_forecast_vs_actual.png"), dpi=300)
        plt.close(fig)

    except Exception as e:
        print(f"ERROR generating forecast vs actual plot for period {period}: {e}"); traceback.print_exc()

def plot_plant_data(plant_name, plant_df_expected, plant_df_actual, market_data_full, opportunity_cost, plot_output_dir):
    """
    Plots a simplified, clearer operational analysis for a single plant,
    focusing on electricity consumption, deviation, and market price with opportunity cost.
    """
    if plant_df_expected.empty and plant_df_actual.empty:
        print(f"No operational data to plot for {plant_name}.")
        return

    plot_df = pd.DataFrame()
    if not plant_df_expected.empty:
        plot_df = plant_df_expected[['Timestamp', 'Expected_Operation_Level', 'Expected_Absolute_Electricity_MWh']].copy()

    if not plant_df_actual.empty:
        actual_df_temp = plant_df_actual[['Timestamp', 'Actual_Operation_Level', 'Actual_Absolute_Electricity_MWh']].copy()
        if not plot_df.empty:
            plot_df = pd.merge(plot_df, actual_df_temp, on='Timestamp', how='outer')
        else:
            plot_df = actual_df_temp

    plot_df.sort_values('Timestamp', inplace=True)
    plot_df.ffill(inplace=True)
    plot_df.fillna(0, inplace=True)

    plot_df['Op_Deviation'] = plot_df['Actual_Operation_Level'] - plot_df['Expected_Operation_Level']

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 16), sharex=True,
                                        gridspec_kw={'height_ratios': [2, 1, 1.5]})
    fig.suptitle(f"Operational Analysis for {plant_name}", fontsize=18, y=0.98)

    ax1.set_title("Operational Level (tons/hr)", fontsize=12)
    ax1.fill_between(plot_df['Timestamp'], 0, plot_df['Expected_Operation_Level'],
                     color='darkorange', alpha=0.3, label='Expected Operation Level')
    ax1.plot(plot_df['Timestamp'], plot_df['Actual_Operation_Level'],
             label='Actual Operation Level', color='firebrick', linewidth=1.5)
    ax1.set_ylabel('Operation Level'); ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.set_title("Operational Deviation from Plan (Actual - Expected)", fontsize=12)
    ax2.plot(plot_df['Timestamp'], plot_df['Op_Deviation'],
              label='Operation Deviation', color='firebrick')
    ax2.axhline(0, color='black', linestyle=':', linewidth=1)
    ax2.set_ylabel('Deviation')
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle=':', alpha=0.6)

    ax3.set_title("Actual Market Clearing Price & Opportunity Cost", fontsize=12)
    if not market_data_full.empty:
        min_ts, max_ts = plot_df['Timestamp'].min(), plot_df['Timestamp'].max()
        mcp_plot_df = market_data_full[(market_data_full['Timestamp'] >= min_ts) & (market_data_full['Timestamp'] <= max_ts)]
        ax3.plot(mcp_plot_df['Timestamp'], mcp_plot_df['MCP'],
                  label='MCP (€/MWh)', color='purple', linewidth=1.5)
    else:
        ax3.text(0.5, 0.5, 'Complete market data not available.', ha='center', va='center', transform=ax3.transAxes)

    if opportunity_cost is not None:
        ax3.axhline(y=opportunity_cost, color='green', linestyle='--',
                    label=f'Electricity Opportunity Cost (€{opportunity_cost:.2f}/MWh)')

    ax3.set_ylabel('Price (€/MWh)')
    ax3.legend(loc='upper left')
    ax3.grid(True, linestyle=':', alpha=0.6)

    ax3.set_xlabel('Timestamp')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot_filename = os.path.join(plot_output_dir, f"{plant_name.replace(' ', '_')}_operation_analysis.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)

def plot_storage_data(material_name, material_df_expected, material_df_actual, plot_output_dir):
    """Plots the hourly storage level for a single material."""
    if material_df_expected.empty and material_df_actual.empty:
        return

    fig, ax = plt.subplots(figsize=(18, 8))
    fig.suptitle(f"Hourly Storage Level for {material_name}", fontsize=16)

    if not material_df_expected.empty:
        ax.plot(material_df_expected['Timestamp'], material_df_expected['Expected_Storage_Level_Tons'],
                label=f'Expected {material_name} Level (tons)', color='teal', linewidth=1.5, alpha=0.8)
    if not material_df_actual.empty:
        ax.plot(material_df_actual['Timestamp'], material_df_actual['Actual_Storage_Level_Tons'],
                label=f'Actual {material_name} Level (tons)', color='brown', linestyle='--', linewidth=1.5, alpha=0.8)

    storage_meta = STORAGE_DEFINITIONS.get(material_name, {})
    min_level = storage_meta.get('Min_Level'); max_level = storage_meta.get('Max_Level')
    if pd.notna(min_level): ax.axhline(y=min_level, color='red', linestyle=':', label=f'Min Level ({min_level:,.0f} tons)', linewidth=1)
    if pd.notna(max_level) and max_level < 1e8: ax.axhline(y=max_level, color='green', linestyle=':', label=f'Max Level ({max_level:,.0f} tons)', linewidth=1)

    ax.set_ylabel(f'{material_name} Level (tons)'); ax.legend(loc='upper left'); ax.grid(True, linestyle=':', alpha=0.7)
    ax.set_xlabel('Timestamp'); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M')); plt.xticks(rotation=45, ha='right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = os.path.join(plot_output_dir, f"{material_name.replace(' ', '_')}_storage_analysis.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)

def plot_cumulative_steel_production(hrs_df_expected, hrs_df_actual, market_data_full, opportunity_cost, config, plot_output_dir):
    """
    Plots cumulative Hot Rolled Steel production and market prices in two separate,
    properly scaled subplots for clarity, with an opportunity cost overlay.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle('Hot Rolled Steel Production and Market Price Analysis', fontsize=16, y=0.99)

    # --- Subplot 1: Cumulative Hot Rolled Steel Production ---
    ax1.set_title('Cumulative Hot Rolled Steel Production', fontsize=12)
    plot_something = False

    if not hrs_df_expected.empty:
        df_exp = hrs_df_expected.sort_values(by='end_date').copy()
        df_exp['cumulative'] = df_exp['total_expected_hot_rolled_steel_tons'].cumsum()
        ax1.plot(df_exp['end_date'], df_exp['cumulative'], label='Cumulative Expected Steel (tons)', color='blue', marker='o', linestyle='-', markersize=4, linewidth=2, alpha=0.8)
        plot_something = True

    if not hrs_df_actual.empty:
        df_act = hrs_df_actual.sort_values(by='end_date').copy()
        df_act['cumulative'] = df_act['total_actual_hot_rolled_steel_tons'].cumsum()
        ax1.plot(df_act['end_date'], df_act['cumulative'], label='Cumulative Actual Steel (tons)', color='green', marker='x', linestyle='--', markersize=5, linewidth=2, alpha=0.8)
        plot_something = True

    sim_params = config.get('simulation', {})
    plant_params = config.get('steel_plant', {})
    sim_start_date_str = sim_params.get('start_date')
    sim_end_date_str = sim_params.get('end_date')
    target_hrs_per_day = plant_params.get('target_hot_rolled_steel_per_day', 0)

    if target_hrs_per_day > 0 and sim_start_date_str and sim_end_date_str:
        sim_start_date = pd.to_datetime(sim_start_date_str).tz_localize(None)
        sim_end_date = pd.to_datetime(sim_end_date_str).tz_localize(None)
        target_dates = pd.date_range(start=sim_start_date, end=sim_end_date, freq='D')
        cumulative_target = [((td.date() - sim_start_date.date()).days + 1) * target_hrs_per_day for td in target_dates]
        ax1.plot(target_dates, cumulative_target, label='Cumulative Target Steel (tons)', color='red', linestyle=':', linewidth=2.5)
        plot_something = True

    ax1.set_ylabel('Cumulative Hot Rolled Steel (tons)')
    if not plot_something:
        ax1.text(0.5, 0.5, 'No Hot Rolled Steel production data to plot.', ha='center', va='center', transform=ax1.transAxes)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.7)

    # --- Subplot 2: Market Clearing Price & Opportunity Cost ---
    ax2.set_title('Market Clearing Price & Opportunity Cost', fontsize=12)
    if not market_data_full.empty:
        min_ts = hrs_df_expected['start_date'].min() if not hrs_df_expected.empty else (hrs_df_actual['start_date'].min() if not hrs_df_actual.empty else None)
        max_ts = hrs_df_expected['end_date'].max() if not hrs_df_expected.empty else (hrs_df_actual['end_date'].max() if not hrs_df_actual.empty else None)

        if pd.notna(min_ts) and pd.notna(max_ts):
            mcp_plot_df = market_data_full[(market_data_full['Timestamp'] >= min_ts) & (market_data_full['Timestamp'] <= max_ts)]
            ax2.plot(mcp_plot_df['Timestamp'], mcp_plot_df['MCP'], color='purple', linewidth=1.5, label='Actual MCP (€/MWh)')
    else:
        ax2.text(0.5, 0.5, 'Complete market data not available.', ha='center', va='center', transform=ax2.transAxes)

    if opportunity_cost is not None:
        ax2.axhline(y=opportunity_cost, color='green', linestyle='--',
                    label=f'Electricity Opportunity Cost (€{opportunity_cost:.2f}/MWh)')

    ax2.set_ylabel('Price (€/MWh)')
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle=':', alpha=0.7)

    ax2.set_xlabel('Date', fontsize=12)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    plot_filename = os.path.join(plot_output_dir, "cumulative_steel_production_analysis.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)

def plot_total_electricity_overview(consumption_data_exp, accepted_bids_df, consumption_data_act, config, plot_output_dir):
    """Plots a simplified overview of total hourly electricity consumption components."""
    fig, ax = plt.subplots(figsize=(18, 8))
    fig.suptitle('Total Hourly Electricity Overview', fontsize=16)

    baseload_mw = config.get('steel_plant', {}).get('baseload_mw', 0.0)

    # Ensure dataframes are not empty before proceeding
    if accepted_bids_df.empty and consumption_data_act.empty and consumption_data_exp.empty:
        print("No electricity data available to plot.")
        plt.close(fig)
        return

    plot_df = pd.DataFrame()
    if not accepted_bids_df.empty:
        actual_cleared = accepted_bids_df.groupby('Date')['Accepted MW'].sum().reset_index().rename(columns={'Date': 'Timestamp'})
        plot_df = actual_cleared

    if not consumption_data_act.empty:
        actual_consumed_flex = consumption_data_act.groupby('Timestamp')['Actual_Absolute_Electricity_MWh'].sum().reset_index()
        if not plot_df.empty:
            plot_df = pd.merge(plot_df, actual_consumed_flex, on='Timestamp', how='outer')
        else:
            plot_df = actual_consumed_flex

    if not consumption_data_exp.empty:
        expected_consumed_flex = consumption_data_exp.groupby('Timestamp')['Expected_Absolute_Electricity_MWh'].sum().reset_index()
        if not plot_df.empty:
            plot_df = pd.merge(plot_df, expected_consumed_flex, on='Timestamp', how='outer')
        else:
            plot_df = expected_consumed_flex

    plot_df.fillna(0, inplace=True)
    plot_df.sort_values('Timestamp', inplace=True)

    if 'Accepted MW' in plot_df.columns:
        ax.plot(plot_df['Timestamp'], plot_df['Accepted MW'], label='Total Cleared from Market', color='black', marker='.', markersize=3, alpha=0.9, linewidth=1.5)

    if 'Actual_Absolute_Electricity_MWh' in plot_df.columns:
        plot_df['Total_Actual_Consumed'] = plot_df['Actual_Absolute_Electricity_MWh'] + baseload_mw
        ax.plot(plot_df['Timestamp'], plot_df['Total_Actual_Consumed'], label='Total Actual Consumption', color='green', linestyle='-', alpha=0.9, linewidth=1.5)

    if 'Expected_Absolute_Electricity_MWh' in plot_df.columns:
        plot_df['Total_Expected_Consumed'] = plot_df['Expected_Absolute_Electricity_MWh'] + baseload_mw
        ax.plot(plot_df['Timestamp'], plot_df['Total_Expected_Consumed'], label='Total Expected Consumption', color='dodgerblue', linestyle='--', alpha=0.8, linewidth=1.5)

    ax.set_xlabel('Timestamp'); ax.set_ylabel('Total Electricity (MWh)')
    ax.legend(loc='upper left'); ax.grid(True, linestyle=':', alpha=0.7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M')); plt.xticks(rotation=45, ha='right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = os.path.join(plot_output_dir, "total_electricity_overview.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)

def plot_unused_cleared_energy(unused_energy_df, plot_output_dir):
    """Plots the total unused energy per re-dispatch period."""
    if unused_energy_df is None or unused_energy_df.empty:
        return

    fig, ax = plt.subplots(figsize=(15, 7))
    fig.suptitle('Total Unused Cleared Energy During Re-dispatch', fontsize=16)
    plot_df = unused_energy_df.sort_values('start_date')
    ax.bar(plot_df['start_date'], plot_df['total_period_unused_mwh'], label='Unused Cleared Energy (MWh)', color='salmon', width=timedelta(days=0.8))
    ax.set_xlabel('Period Start Date'); ax.set_ylabel('Unused Energy (MWh)')
    ax.legend(loc='upper right'); ax.grid(True, axis='y', linestyle=':', alpha=0.7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45, ha='right'); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = os.path.join(plot_output_dir, "unused_cleared_energy_periodic.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)

def plot_forecast_with_scenarios_window(period_info, config, main_output_dir, plot_output_dir):
    """
    Plots the forecast for a window along with the generated price scenarios
    using a clearer "fan chart" visualization with percentile bands.
    """
    if period_info is None or pd.isna(period_info.get('period')) or pd.isna(period_info.get('start_date')):
        print("Skipping scenario plot due to missing period_info data.")
        return

    period = int(period_info['period'])
    window_start_date = pd.to_datetime(period_info['start_date']).tz_localize(None)
    actual_horizon = get_actual_planning_horizon_for_window(window_start_date, config)

    # Load forecast data
    forecast_load_dir = os.path.join(main_output_dir, 'forecast_outputs_per_window')
    forecast_file = os.path.join(forecast_load_dir, f"forecast_p{period}_{window_start_date.strftime('%Y%m%d')}_h{actual_horizon}d.csv")
    if not os.path.exists(forecast_file):
        print(f"Warning: Forecast file not found for scenario plot: {forecast_file}")
        return
    forecast_df = pd.read_csv(forecast_file)
    forecast_prices = forecast_df['Forecast_Price'].values

    # Load scenario data
    sim_out_dir_name = os.path.basename(config.get('simulation', {}).get('output_dir', 'results').rstrip('/\\'))
    scenario_dir = os.path.join(main_output_dir, f'scenarios_{sim_out_dir_name}')
    scenario_file = os.path.join(scenario_dir, f"scenarios_p{period}_{window_start_date.strftime('%Y%m%d')}.csv")
    if not os.path.exists(scenario_file):
        print(f"Warning: Scenario file not found: {scenario_file}")
        return
    scenarios_df = pd.read_csv(scenario_file)
    hour_cols = [col for col in scenarios_df.columns if col.startswith('Hour_')]
    scenarios = scenarios_df[hour_cols].values

    # --- Start Plotting ---
    fig, ax = plt.subplots(figsize=(18, 9))

    num_hours = scenarios.shape[1]
    timestamps = pd.to_datetime([window_start_date + timedelta(hours=i) for i in range(num_hours)])

    ax.plot(timestamps, scenarios.T, color='grey', alpha=0.1, linewidth=0.5)
    ax.plot([], [], color='grey', alpha=0.3, label='Individual Scenarios')

    p10 = np.percentile(scenarios, 10, axis=0)
    p25 = np.percentile(scenarios, 25, axis=0)
    p50 = np.percentile(scenarios, 50, axis=0)
    p75 = np.percentile(scenarios, 75, axis=0)
    p90 = np.percentile(scenarios, 90, axis=0)

    ax.fill_between(timestamps, p10, p90, color='dodgerblue', alpha=0.2, label='10th-90th Percentile')
    ax.fill_between(timestamps, p25, p75, color='dodgerblue', alpha=0.4, label='25th-75th Percentile (IQR)')

    ax.plot(timestamps, p50, color='red', linestyle='-', linewidth=2.5, label='Median Scenario (P50)')
    ax.plot(timestamps, forecast_prices, color='black', linestyle='--', linewidth=2.5, label='Original Forecast (Mean)')

    ax.set_title(f'Price Scenarios and Confidence Intervals (Period {period})', fontsize=16)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Price (€/MWh)', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.6)
    y_min_val = min(np.min(p10), np.min(forecast_prices))
    y_max_val = max(np.max(p90), np.max(forecast_prices))
    ax.set_ylim(bottom=y_min_val - 0.1 * abs(y_min_val), top=y_max_val + 0.1 * abs(y_max_val))

    fig.autofmt_xdate()
    plt.tight_layout()

    plot_filename = os.path.join(plot_output_dir, f"period_{period}_forecast_with_scenarios.png")
    try:
        plt.savefig(plot_filename, dpi=300)
    except Exception as e:
        print(f"Error saving scenario plot: {e}")
    plt.close(fig)

def plot_bidding_curves_for_day(day_bids_df, plot_date, plot_output_dir):
    """Plots the bidding demand curves for several hours within a single day."""
    if day_bids_df.empty: return

    fig, ax = plt.subplots(figsize=(15, 9))
    hours_in_data = sorted(day_bids_df['Date'].dt.hour.unique())
    # Plot up to 6 representative hours for clarity
    hours_to_plot = hours_in_data[::(len(hours_in_data)//6 + 1)] if len(hours_in_data) > 6 else hours_in_data

    colors = plt.cm.viridis(np.linspace(0, 1, len(hours_to_plot)))

    for i, hour in enumerate(hours_to_plot):
        bids_hour = day_bids_df[day_bids_df['Date'].dt.hour == hour].copy()
        if bids_hour.empty: continue
        bids_hour.sort_values(by='Bid Price', ascending=False, inplace=True)
        bids_hour['Cumulative Quantity (MW)'] = bids_hour['Bid Quantity (MW)'].cumsum()

        ax.step([0] + bids_hour['Cumulative Quantity (MW)'].tolist(),
                [bids_hour['Bid Price'].iloc[0]] + bids_hour['Bid Price'].tolist(),
                where='post', label=f'Hour {hour:02d}:00', color=colors[i], linewidth=1.5)

    ax.set_title(f"Demand Curves for Implementation Day ({plot_date.strftime('%Y-%m-%d')})", fontsize=14)
    ax.set_xlabel("Cumulative Quantity (MW)"); ax.set_ylabel("Bid Price (€/MWh)")
    ax.legend(loc='best'); ax.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plot_filename = os.path.join(plot_output_dir, f"bidding_curves_{plot_date.strftime('%Y%m%d')}.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)

def main():
    """
    Main function to orchestrate loading data and generating all plots.
    MODIFIED to run directly without command-line arguments.
    """
    # --- CHANGE: Hardcode the default paths here ---
    # Assumes a project structure where 'config_steel.yaml' is in the root
    # and the output directory is as specified in that config.
    default_config_file = 'config.yaml'

    config = load_config(default_config_file)
    if config is None:
        print(f"FATAL: Could not load the default config file: {default_config_file}")
        sys.exit(1)

    # Get output directory from the loaded config
    output_dir_arg = config.get('simulation', {}).get('output_dir')
    if not output_dir_arg:
        print("FATAL: 'output_dir' not specified in the simulation config.")
        sys.exit(1)

    print(f"Starting visualization process for STEEL PLANT model outputs in: {output_dir_arg}")

    loaded_data, market_data_full = load_steel_plant_data(output_dir_arg, config)
    plot_main_dir = os.path.join(output_dir_arg, "simulation_plots_steel_plant")
    os.makedirs(plot_main_dir, exist_ok=True)

    # --- Plot Plant Operations ---
    consumption_data_exp = loaded_data.get("consumption_data_expected", pd.DataFrame())
    consumption_data_act = loaded_data.get("consumption_data_actual", pd.DataFrame())
    plant_plot_dir = os.path.join(plot_main_dir, "plant_operations")
    os.makedirs(plant_plot_dir, exist_ok=True)

    # --- ADAPTATION: Read opportunity cost from steel_plant section ---
    opportunity_cost = config.get('steel_plant', {}).get('opportunity_cost_mwh_shortfall')
    if opportunity_cost is not None:
        print(f"INFO: Found opportunity_cost_mwh_shortfall in config: {opportunity_cost}")
    else:
        print("WARNING: 'opportunity_cost_mwh_shortfall' not found under 'steel_plant' in the config file. The line will not be plotted.")

    all_plants = set(consumption_data_exp['Plant_Name'].unique()) | set(consumption_data_act['Plant_Name'].unique())
    for plant in sorted(list(all_plants)):
        plant_df_exp = consumption_data_exp[consumption_data_exp['Plant_Name'] == plant]
        plant_df_act = consumption_data_act[consumption_data_act['Plant_Name'] == plant]
        plot_plant_data(plant, plant_df_exp, plant_df_act, market_data_full, opportunity_cost, plant_plot_dir)

    # --- Plot Storage Levels ---
    storage_data_exp = loaded_data.get("storage_data_hourly_expected", pd.DataFrame())
    storage_data_act = loaded_data.get("storage_data_hourly_actual", pd.DataFrame())
    storage_plot_dir = os.path.join(plot_main_dir, "storage_levels")
    os.makedirs(storage_plot_dir, exist_ok=True)
    all_materials = set(storage_data_exp['Material'].unique()) | set(storage_data_act['Material'].unique())
    for material in sorted(list(all_materials)):
        material_df_exp = storage_data_exp[storage_data_exp['Material'] == material]
        material_df_act = storage_data_act[storage_data_act['Material'] == material]
        plot_storage_data(material, material_df_exp, material_df_act, storage_plot_dir)

    # --- Plot Cumulative Hot Rolled Steel Production ---
    hrs_df_exp = loaded_data.get("hot_rolled_steel_prod_file_expected", pd.DataFrame())
    hrs_df_act = loaded_data.get("hot_rolled_steel_prod_file_actual", pd.DataFrame())
    prod_summary_dir = os.path.join(plot_main_dir, "production_summary")
    os.makedirs(prod_summary_dir, exist_ok=True)
    plot_cumulative_steel_production(hrs_df_exp, hrs_df_act, market_data_full, opportunity_cost, config, prod_summary_dir)

    # --- Plot Electricity Overview & Unused Energy ---
    elec_dir = os.path.join(plot_main_dir, "electricity_analysis")
    os.makedirs(elec_dir, exist_ok=True)
    plot_total_electricity_overview(consumption_data_exp, loaded_data.get('accepted_bids_df_full'), consumption_data_act, config, elec_dir)
    plot_unused_cleared_energy(loaded_data.get('unused_energy_df'), elec_dir)

    # --- Plot Windowed Forecasts, Scenarios, and Bidding Curves ---
    forecast_periods_info = loaded_data.get('forecast_periods_info')
    if forecast_periods_info is not None and not forecast_periods_info.empty:
        forecast_plot_dir = os.path.join(plot_main_dir, "forecast_evaluations")
        scenario_plot_dir = os.path.join(plot_main_dir, "scenario_visualizations")
        bidding_curve_dir = os.path.join(plot_main_dir, "bidding_curves")
        os.makedirs(forecast_plot_dir, exist_ok=True)
        os.makedirs(scenario_plot_dir, exist_ok=True)
        os.makedirs(bidding_curve_dir, exist_ok=True)

        for _, row in forecast_periods_info.iterrows():
            plot_forecast_vs_actual_window(row, config, market_data_full, output_dir_arg, forecast_plot_dir)
            plot_forecast_with_scenarios_window(row, config, output_dir_arg, scenario_plot_dir)

        all_bids_df = loaded_data.get('all_bids_made_df')
        if all_bids_df is not None and not all_bids_df.empty:
            for day, group in all_bids_df.groupby(all_bids_df['Date'].dt.date):
                plot_bidding_curves_for_day(group, pd.to_datetime(day), bidding_curve_dir)

    print("\nVisualization process complete for Steel Plant.")


if __name__ == "__main__":
    # --- REMOVED argparse to allow direct running from VS Code ---
    main()