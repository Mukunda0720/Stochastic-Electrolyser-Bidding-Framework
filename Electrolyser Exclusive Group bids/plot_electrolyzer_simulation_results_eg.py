# plot_electrolyzer_simulation_results_eg.py
# V8 - Added scenario visualization "fan charts" from the other plotting script.

"""
Generates plots for visualizing electrolyzer EXCLUSIVE GROUP BID simulation results including:
- A simplified plant analysis focusing on electricity consumption, deviation, and market price.
- A comprehensive profile plot showing the verified accepted profile (green), the top-ranked bid (blue), 
  and all other submitted bids (grey) against the market price.
- Price scenario "fan charts" for clear percentile visualization.
- Actual storage levels (e.g., for Water).
- Price forecasts vs. actuals.
- Cumulative hydrogen production (actual vs. target) with market prices.

Usage (VS Code / Terminal):
    1. Place this script in the same directory as your config file (e.g., 'config_eg.yaml').
    2. Run the script. It will automatically find the config and the output directory listed within it.
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
import glob

try:
    from electrolyzer_definitions import PLANT_DEFINITIONS, ABSOLUTE_PLANT_LIMITS, STORAGE_DEFINITIONS
    from ml_forecaster import ElectricityPriceForecaster
    from ml_scenario_generator import ScenarioGenerator
except ImportError as e:
    print(f"ERROR: Could not import necessary modules: {e}")
    print("Ensure electrolyzer_definitions.py, ml_forecaster.py, and ml_scenario_generator.py are accessible.")
    sys.exit(1)

def load_config(config_file_path):
    """Loads the YAML configuration file."""
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

# --- NEW FUNCTION (from other script) ---
def get_actual_planning_horizon_for_window(window_start_date, config):
    """Calculates the true planning horizon, accounting for the simulation end date."""
    sim_config = config.get('simulation', {})
    planning_horizon_days_config = sim_config.get('planning_horizon', 2)
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

def load_eg_model_data(output_dir, config):
    """Loads all necessary data files from an EG simulation output directory."""
    try:
        dir_from_config = config.get('simulation',{}).get('output_dir','unknown')
        output_dir_basename = os.path.basename(dir_from_config.rstrip(os.sep))
    except Exception:
        output_dir_basename = os.path.basename(output_dir.rstrip(os.sep))

    print(f"DEBUG: Using output file prefix: '{output_dir_basename}'")

    files_to_try = {
        "consumption_data_actual": os.path.join(output_dir, f"{output_dir_basename}_plant_operations_details_actual_redispatch.xlsx"),
        "storage_data_hourly_actual": os.path.join(output_dir, f"{output_dir_basename}_hourly_storage_levels_details_actual_redispatch.xlsx"),
        "hydrogen_prod_df_actual": os.path.join(output_dir, f"{output_dir_basename}_hydrogen_produced_periodical_actual.xlsx"),
        "accepted_bids_df_full": os.path.join(output_dir, f"{output_dir_basename}_accepted_eg_bid_details.xlsx"),
        "submitted_profiles_df": os.path.join(output_dir, f"{output_dir_basename}_all_submitted_profiles_detailed.xlsx"),
        "forecast_errors_file": os.path.join(output_dir, f"{output_dir_basename}_forecast_errors.xlsx"),
        "market_clearing_summary": os.path.join(output_dir, f"{output_dir_basename}_market_clearing_summary_eg.xlsx")
    }

    market_input_file_from_config = config.get('simulation', {}).get('market_input_file', None)
    loaded_data = {}

    for key, filepath in files_to_try.items():
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

    mcp_data_actual_unique = pd.DataFrame()
    accepted_bids_df_full = loaded_data.get("accepted_bids_df_full", pd.DataFrame())
    if not accepted_bids_df_full.empty and 'Date' in accepted_bids_df_full.columns and 'MCP' in accepted_bids_df_full.columns:
        mcp_data_actual_unique = accepted_bids_df_full.drop_duplicates(subset=['Date'])[['Date', 'MCP']].rename(columns={'Date': 'Timestamp'})

    market_data_full = pd.DataFrame()
    if market_input_file_from_config:
        try:
            # Construct absolute path for market input file if it's relative
            market_input_file_abs = market_input_file_from_config
            if not os.path.isabs(market_input_file_abs):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                market_input_file_abs = os.path.abspath(os.path.join(script_dir, market_input_file_from_config))

            if os.path.exists(market_input_file_abs):
                target_var = config.get('forecast', {}).get('target_variable', 'Price')
                temp_forecaster = ElectricityPriceForecaster({'target_variable': target_var})
                market_data_full = temp_forecaster.load_data(market_input_file_abs).rename(columns={target_var: 'MCP', 'Date': 'Timestamp'})
                print(f"Successfully loaded full market data from {market_input_file_abs}")
            else:
                print(f"Warning: Market input file specified but not found: {market_input_file_abs}")

        except Exception as e:
            print(f"Error loading full market data: {e}")
    else:
        print("Warning: Full market data file not specified in config. Cannot plot a complete MCP line.")


    loaded_data["mcp_data_actual_unique"] = mcp_data_actual_unique
    loaded_data["market_data_full"] = market_data_full
    return loaded_data

def plot_plant_data_eg(plant_name, plant_df_actual, market_data_full, opportunity_cost, plot_output_dir):
    if plant_df_actual.empty:
        print(f"No actual operational data to plot for {plant_name}.")
        return

    plot_df = plant_df_actual[['Timestamp', 'Actual_Absolute_Electricity_MWh']].copy()
    plot_df.sort_values('Timestamp', inplace=True)
    plot_df.fillna(0, inplace=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1.5]})
    fig.suptitle(f"Operational Analysis for {plant_name} (Exclusive Group)", fontsize=18, y=0.98)

    ax1.set_title("Actual Electricity Consumption", fontsize=12)
    ax1.plot(plot_df['Timestamp'], plot_df['Actual_Absolute_Electricity_MWh'],
             label='Actual Electricity (MWh)', color='firebrick', linewidth=1.5)
    ax1.set_ylabel('Electricity (MWh)'); ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.set_title("Actual Market Clearing Price & Opportunity Cost", fontsize=12)
    if not market_data_full.empty:
        min_ts, max_ts = plot_df['Timestamp'].min(), plot_df['Timestamp'].max()
        mcp_plot_df = market_data_full[(market_data_full['Timestamp'] >= min_ts) & (market_data_full['Timestamp'] <= max_ts)]
        ax2.plot(mcp_plot_df['Timestamp'], mcp_plot_df['MCP'],
                  label='MCP (€/MWh)', color='purple', linewidth=1.5)
    else:
        if not plant_df_actual.empty and 'MCP' in plant_df_actual.columns:
             ax2.plot(plant_df_actual['Timestamp'], plant_df_actual['MCP'],
                  label='MCP (€/MWh)', color='purple', linewidth=1.5, linestyle=':', marker='.')
        else:
             ax2.text(0.5, 0.5, 'Market price data not available.', ha='center', va='center', transform=ax2.transAxes)

    if opportunity_cost is not None:
        ax2.axhline(y=opportunity_cost, color='green', linestyle='--',
                    label=f'Opportunity Cost (€{opportunity_cost:.2f}/MWh)')

    ax2.set_ylabel('Price (€/MWh)')
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle=':', alpha=0.6)

    ax2.set_xlabel('Timestamp')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot_filename = os.path.join(plot_output_dir, f"{plant_name.replace(' ', '_')}_operation_analysis_eg.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)


def plot_storage_data_eg(material_name, material_df_actual, plot_output_dir):
    if material_df_actual.empty: return

    storage_meta = STORAGE_DEFINITIONS.get(material_name, {})
    fig, ax = plt.subplots(figsize=(18, 8)); fig.suptitle(f"Actual Hourly Storage Level for {material_name} (EG)", fontsize=16)
    ax.plot(material_df_actual['Timestamp'], material_df_actual['Actual_Storage_Level_Tons'], label=f'Actual {material_name} Level (tons)', color='brown', linestyle='-', linewidth=1.5)
    min_level, max_level = storage_meta.get('Min_Level'), storage_meta.get('Max_Level')
    if pd.notna(min_level): ax.axhline(y=min_level, color='red', linestyle=':', label=f'Min Level ({min_level:.2f})')
    if pd.notna(max_level) and max_level < 1e8: ax.axhline(y=max_level, color='green', linestyle=':', label=f'Max Level ({max_level:.2f})')
    ax.set_ylabel(f'{material_name} Level (tons)'); ax.legend(); ax.grid(True, alpha=0.5); ax.set_xlabel('Timestamp'); fig.autofmt_xdate()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(os.path.join(plot_output_dir, f"{material_name.replace(' ', '_')}_storage_level_analysis_eg.png"), dpi=300); plt.close(fig)

def plot_cumulative_hydrogen_production_eg(h2_df_actual, mcp_df_actual_unique, config, plot_output_dir):
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_title('Cumulative Hydrogen Production & Market Price (EG)', fontsize=16)

    if h2_df_actual.empty:
        print("Warning: Hydrogen production data is empty. Cannot plot cumulative production.")
        ax.text(0.5, 0.5, 'Hydrogen production data not available.', ha='center', va='center', transform=ax.transAxes)
        plt.savefig(os.path.join(plot_output_dir, "cumulative_hydrogen_production_analysis_eg.png"), dpi=300)
        plt.close(fig)
        return

    sim_params = config.get('simulation', {})
    plant_params = config.get('electrolyzer_plant', {})
    sim_start_date = pd.to_datetime(sim_params.get('start_date')).tz_localize(None)
    sim_end_date = pd.to_datetime(sim_params.get('end_date')).tz_localize(None)
    target_h2_per_day = plant_params.get('target_hydrogen_per_day', 0)

    sort_col = 'end_date' if 'end_date' in h2_df_actual.columns else 'start_date'

    actual_h2_plot_df = h2_df_actual.sort_values(by=sort_col).copy()
    x_axis_data = actual_h2_plot_df[sort_col]
    ax.set_xlabel('Date')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    actual_h2_plot_df['cumulative_actual_h2'] = actual_h2_plot_df['total_actual_hydrogen_kg'].cumsum()
    ax.plot(x_axis_data, actual_h2_plot_df['cumulative_actual_h2'], label='Cumulative Actual Hydrogen (kg)', color='green', marker='x', linestyle='-', linewidth=2, zorder=10)
    ax.set_ylabel('Cumulative Hydrogen (kg)', color='green')
    ax.tick_params(axis='y', labelcolor='green')

    if target_h2_per_day > 0 and pd.notna(sim_start_date) and pd.notna(sim_end_date):
        target_dates = pd.date_range(start=sim_start_date, end=sim_end_date, freq='D')
        cumulative_target = [(d.date() - sim_start_date.date()).days * target_h2_per_day for d in target_dates]
        ax.plot(target_dates, cumulative_target, label='Cumulative Target Hydrogen (kg)', color='red', linestyle=':', linewidth=2, zorder=5)

    ax2 = ax.twinx()
    if not mcp_df_actual_unique.empty:
        ax2.plot(mcp_df_actual_unique['Timestamp'], mcp_df_actual_unique['MCP'],
                 label='Actual MCP (€/MWh)', color='purple', linestyle='--',
                 marker='.', markersize=4, alpha=0.6, zorder=1)
    ax2.set_ylabel('Actual MCP (€/MWh)', color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.grid(False)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left')

    ax.grid(True, alpha=0.5)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_output_dir, "cumulative_hydrogen_production_analysis_eg.png"), dpi=300)
    plt.close(fig)

def plot_total_electricity_overview_eg(consumption_data_actual, accepted_bids_df_full, config, plot_output_dir):
    fig, ax = plt.subplots(figsize=(18, 8))
    fig.suptitle('Total Hourly Electricity Overview (EG)', fontsize=16)

    baseload_mw = config.get('electrolyzer_plant', {}).get('baseload_mw', 0.0)

    actual_consumed_flex = pd.DataFrame()
    if not consumption_data_actual.empty:
        actual_consumed_flex = consumption_data_actual.groupby('Timestamp')['Actual_Absolute_Electricity_MWh'].sum().reset_index()
        actual_consumed_flex.rename(columns={'Actual_Absolute_Electricity_MWh': 'Flex_Plants_Consumed_MWh'}, inplace=True)

    actual_cleared_total = pd.DataFrame()
    if not accepted_bids_df_full.empty:
        actual_cleared_total = accepted_bids_df_full.groupby('Date')['Accepted MW'].sum().reset_index()
        actual_cleared_total.rename(columns={'Date': 'Timestamp', 'Accepted MW': 'Total_Cleared_MWh'}, inplace=True)

    all_timestamps = set(actual_consumed_flex['Timestamp']) | set(actual_cleared_total['Timestamp'])
    if not all_timestamps:
        ax.text(0.5, 0.5, 'No electricity data.', ha='center', va='center', transform=ax.transAxes)
        plt.close(fig); return

    plot_df = pd.DataFrame(sorted(list(all_timestamps)), columns=['Timestamp'])
    if not actual_consumed_flex.empty: plot_df = pd.merge(plot_df, actual_consumed_flex, on='Timestamp', how='left')
    if not actual_cleared_total.empty: plot_df = pd.merge(plot_df, actual_cleared_total, on='Timestamp', how='left')
    plot_df.fillna(0, inplace=True)

    if 'Flex_Plants_Consumed_MWh' in plot_df.columns:
        plot_df['Total_Actual_Consumed_MWh'] = plot_df['Flex_Plants_Consumed_MWh'] + baseload_mw
        ax.plot(plot_df['Timestamp'], plot_df['Total_Actual_Consumed_MWh'], label='Total Actual Consumption', color='green', linestyle='-', alpha=0.9, linewidth=2)

    if 'Total_Cleared_MWh' in plot_df.columns:
        ax.plot(plot_df['Timestamp'], plot_df['Total_Cleared_MWh'], label='Total Cleared from Market', color='black', marker='.', markersize=4, alpha=0.9, linewidth=1.5)

    ax.set_xlabel('Timestamp'); ax.set_ylabel('Total Electricity (MWh)')
    ax.legend(loc='upper left'); ax.grid(True, linestyle=':', alpha=0.7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45, ha='right'); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(plot_output_dir, "total_electricity_overview_eg.png"), dpi=300); plt.close(fig)

def plot_forecast_vs_actual_window(period_info, config, market_data_full, main_output_dir, plot_output_dir):
    if period_info is None: return
    period = int(period_info['period'])
    window_start_date = pd.to_datetime(period_info['start_date']).tz_localize(None)

    forecast_load_dir = os.path.join(main_output_dir, 'forecast_outputs_per_window_eg')
    file_pattern = os.path.join(forecast_load_dir, f"forecast_eg_p{period}_{window_start_date.strftime('%Y%m%d')}_h*d.csv")
    found_files = glob.glob(file_pattern)

    if not found_files:
        print(f"Warning: Forecast file not found for period {period}. Pattern: {file_pattern}"); return

    try:
        forecast_df = pd.read_csv(found_files[0])
        forecast_df['Date'] = pd.to_datetime(forecast_df['Date']).dt.tz_localize(None)

        plt.figure(figsize=(15, 7))
        plt.plot(forecast_df['Date'], forecast_df['Forecast_Price'], label='Forecast', color='blue')
        if not market_data_full.empty:
            actual_df_window = market_data_full[market_data_full['Timestamp'].isin(forecast_df['Date'])]
            plt.plot(actual_df_window['Timestamp'], actual_df_window['MCP'], label='Actual', color='red', linestyle='--')

        plt.title(f'Period {period}: Price Forecast vs Actual'); plt.xlabel('Date'); plt.ylabel('Price (€/MWh)')
        plt.legend(); plt.grid(True, alpha=0.5); plt.tight_layout()
        plt.savefig(os.path.join(plot_output_dir, f"period_{period}_forecast_vs_actual_eg.png"), dpi=300); plt.close()
    except Exception as e:
        print(f"ERROR plotting forecast for period {period}: {e}")

# --- NEW FUNCTION (from other script, adapted for EG) ---
def plot_forecast_with_scenarios_window_eg(period_info, config, main_output_dir, plot_output_dir):
    """Generates a 'fan chart' of price scenarios against the mean forecast for a given window."""
    if period_info is None or pd.isna(period_info.get('period')) or pd.isna(period_info.get('start_date')):
        print("Skipping scenario plot due to missing period_info data.")
        return
        
    period = int(period_info['period'])
    window_start_date = pd.to_datetime(period_info['start_date']).tz_localize(None)
    actual_horizon = get_actual_planning_horizon_for_window(window_start_date, config)
    
    # Load Forecast
    forecast_load_dir = os.path.join(main_output_dir, 'forecast_outputs_per_window_eg')
    forecast_pattern = os.path.join(forecast_load_dir, f"forecast_eg_p{period}_{window_start_date.strftime('%Y%m%d')}_h*d.csv")
    forecast_files = glob.glob(forecast_pattern)
    if not forecast_files: 
        print(f"Warning: Forecast file not found for EG scenario plot: {forecast_pattern}")
        return
    forecast_df = pd.read_csv(forecast_files[0])
    forecast_prices = forecast_df['Forecast_Price'].values
    
    # Load Scenarios
    sim_out_dir_name = os.path.basename(config.get('simulation', {}).get('output_dir', 'results_eg').rstrip('/\\'))
    scenario_dir = os.path.join(main_output_dir, f'scenarios_{sim_out_dir_name}')
    scenario_file = os.path.join(scenario_dir, f"scenarios_eg_p{period}_{window_start_date.strftime('%Y%m%d')}.csv")
    if not os.path.exists(scenario_file):
        print(f"Warning: Scenario file not found: {scenario_file}")
        return
    scenarios_df = pd.read_csv(scenario_file)
    hour_cols = [col for col in scenarios_df.columns if col.startswith('Hour_')]
    scenarios = scenarios_df[hour_cols].values
    
    # Plotting
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
    ax.plot(timestamps, forecast_prices[:len(timestamps)], color='black', linestyle='--', linewidth=2.5, label='Original Forecast (Mean)')

    ax.set_title(f'Price Scenarios and Confidence Intervals (Period {period}, EG)', fontsize=16)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Price (€/MWh)', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.6)
    y_min_val = min(np.min(p10), np.min(forecast_prices))
    y_max_val = max(np.max(p90), np.max(forecast_prices))
    ax.set_ylim(bottom=y_min_val - 0.1 * abs(y_min_val), top=y_max_val + 0.1 * abs(y_max_val))

    fig.autofmt_xdate()
    plt.tight_layout()
    
    plot_filename = os.path.join(plot_output_dir, f"period_{period}_forecast_with_scenarios_eg.png")
    try:
        plt.savefig(plot_filename, dpi=300)
    except Exception as e:
        print(f"Error saving scenario plot: {e}")
    plt.close(fig)

def plot_profile_analysis_with_context(period_info, accepted_bids_df, submitted_profiles_df, config, plot_output_dir):
    if period_info is None or pd.isna(period_info.get('period')):
        return

    period = int(period_info['period'])
    period_start_date = pd.to_datetime(period_info['start_date'])
    impl_period_days = config.get('simulation', {}).get('implementation_period', 1)
    period_end_date = period_start_date + timedelta(days=impl_period_days, seconds=-1)
    num_hours_to_plot = impl_period_days * 24

    accepted_data_for_period = accepted_bids_df[
        (accepted_bids_df['Date'] >= period_start_date) &
        (accepted_bids_df['Date'] <= period_end_date)
    ].copy()

    fig, ax1 = plt.subplots(figsize=(18, 9))
    fig.suptitle(f"Profile Analysis for {period_start_date.strftime('%Y-%m-%d')} (Period {period})", fontsize=16)

    period_profiles_submitted = submitted_profiles_df[submitted_profiles_df['period'] == period].copy()

    if accepted_data_for_period.empty:
        ax1.text(0.5, 0.5, 'No profile was accepted for this period.',
                 ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        if not period_profiles_submitted.empty:
             hour_cols_all = sorted([col for col in period_profiles_submitted.columns if col.startswith('Hour_') and col.endswith('_MW')])
             hour_cols_impl = hour_cols_all[:num_hours_to_plot]
             timestamps_impl = pd.to_datetime([period_start_date + timedelta(hours=i) for i in range(len(hour_cols_impl))])
             for _, row in period_profiles_submitted.iterrows():
                 ax1.plot(timestamps_impl, row[hour_cols_impl].values, color='gray', linewidth=1, alpha=0.4)
        plt.savefig(os.path.join(plot_output_dir, f"period_{period}_profile_analysis.png"), dpi=300)
        plt.close(fig)
        return

    accepted_id = accepted_data_for_period['Profile_ID'].iloc[0]
    top_ranked_id = None
    if not period_profiles_submitted.empty and 'rank' in period_profiles_submitted.columns:
        top_ranked_row = period_profiles_submitted.loc[period_profiles_submitted['rank'].idxmin()]
        top_ranked_id = top_ranked_row['profile_id']

    hour_cols_all = sorted([col for col in period_profiles_submitted.columns if col.startswith('Hour_') and col.endswith('_MW')])
    hour_cols_impl = hour_cols_all[:num_hours_to_plot]
    timestamps_impl = pd.to_datetime([period_start_date + timedelta(hours=i) for i in range(len(hour_cols_impl))])

    for _, row in period_profiles_submitted.iterrows():
        if row['profile_id'] != accepted_id and row['profile_id'] != top_ranked_id:
            ax1.plot(timestamps_impl, row[hour_cols_impl].values, color='gray', linewidth=1, alpha=0.4, zorder=2)
    ax1.plot([], [], color='gray', linewidth=1, label='Other Submitted Profiles')

    if top_ranked_id is not None and top_ranked_id != accepted_id:
        top_ranked_row = period_profiles_submitted[period_profiles_submitted['profile_id'] == top_ranked_id].iloc[0]
        label = f'Top-Ranked Profile #{int(top_ranked_id)}'
        ax1.plot(timestamps_impl, top_ranked_row[hour_cols_impl].values,
                 color='dodgerblue', linestyle='--', linewidth=2.5, zorder=5, label=label)

    label = f'Accepted Profile #{int(accepted_id)}'
    if accepted_id == top_ranked_id:
        label += ' (Also Top-Ranked)'
    ax1.plot(accepted_data_for_period['Date'], accepted_data_for_period['Accepted MW'],
             color='green', linewidth=3, label=label, zorder=10)

    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Power (MW)', color='navy')
    ax1.tick_params(axis='y', labelcolor='navy')
    ax1.grid(True, which='major', linestyle=':', linewidth=0.7)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Market Clearing Price (€/MWh)', color='purple')
    ax2.plot(accepted_data_for_period['Date'], accepted_data_for_period['MCP'],
             color='purple', linestyle='--', marker='o', label='Actual MCP (€/MWh)')
    ax2.tick_params(axis='y', labelcolor='purple')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(plot_output_dir, f"period_{period}_profile_analysis.png"), dpi=300)
    plt.close(fig)


def main(output_dir_arg, config_file_arg):
    """Main function to orchestrate loading data and generating all plots."""
    print(f"Starting EG visualization process for: {output_dir_arg}")
    config = load_config(config_file_arg)
    if config is None: sys.exit(1)

    loaded_data = load_eg_model_data(output_dir_arg, config)
    plot_main_dir = os.path.join(output_dir_arg, "simulation_plots_electrolyzer_eg")
    os.makedirs(plot_main_dir, exist_ok=True)

    plot_dirs = {
        "plant_ops": os.path.join(plot_main_dir, "plant_operations"),
        "storage": os.path.join(plot_main_dir, "storage_levels"),
        "production": os.path.join(plot_main_dir, "production_summary"),
        "electricity": os.path.join(plot_main_dir, "electricity_overview"),
        "forecasts": os.path.join(plot_main_dir, "forecast_evaluations"),
        "profiles": os.path.join(plot_main_dir, "profile_selection_analysis"),
        "scenarios": os.path.join(plot_main_dir, "scenario_visualizations") # <-- ADDED
    }
    for d in plot_dirs.values(): os.makedirs(d, exist_ok=True)

    opportunity_cost = config.get('electrolyzer_plant', {}).get('opportunity_cost_mwh_shortfall')
    if opportunity_cost is not None:
        print(f"INFO: Found opportunity_cost_mwh_shortfall in config: {opportunity_cost}")
    else:
        print("WARNING: 'opportunity_cost_mwh_shortfall' not found under 'electrolyzer_plant' in the config file. The line will not be plotted.")

    if not loaded_data['consumption_data_actual'].empty:
        for name in loaded_data['consumption_data_actual']['Plant_Name'].unique():
            df = loaded_data['consumption_data_actual'][loaded_data['consumption_data_actual']['Plant_Name'] == name]
            plot_plant_data_eg(name, df, loaded_data['market_data_full'], opportunity_cost, plot_dirs["plant_ops"])

    if not loaded_data['storage_data_hourly_actual'].empty:
        for name in loaded_data['storage_data_hourly_actual']['Material'].unique():
            df = loaded_data['storage_data_hourly_actual'][loaded_data['storage_data_hourly_actual']['Material'] == name]
            plot_storage_data_eg(name, df, plot_dirs["storage"])

    plot_cumulative_hydrogen_production_eg(
        loaded_data['hydrogen_prod_df_actual'],
        loaded_data['mcp_data_actual_unique'],
        config,
        plot_dirs["production"]
    )

    plot_total_electricity_overview_eg(loaded_data['consumption_data_actual'], loaded_data['accepted_bids_df_full'], config, plot_dirs["electricity"])

    if 'forecast_errors_file' in loaded_data and not loaded_data['forecast_errors_file'].empty:
        for _, row in loaded_data['forecast_errors_file'].iterrows():
             plot_forecast_vs_actual_window(row, config, loaded_data['market_data_full'], output_dir_arg, plot_dirs["forecasts"])
             # --- ADDED CALL TO NEW FUNCTION ---
             plot_forecast_with_scenarios_window_eg(row, config, output_dir_arg, plot_dirs["scenarios"])


    if 'market_clearing_summary' in loaded_data and not loaded_data['market_clearing_summary'].empty and 'accepted_bids_df_full' in loaded_data and not loaded_data['accepted_bids_df_full'].empty and 'submitted_profiles_df' in loaded_data and not loaded_data['submitted_profiles_df'].empty:
        for _, row in loaded_data['market_clearing_summary'].iterrows():
            plot_profile_analysis_with_context(
                row,
                loaded_data['accepted_bids_df_full'],
                loaded_data['submitted_profiles_df'],
                config,
                plot_dirs["profiles"]
            )
    else:
        print("\nSkipping profile analysis plots: one or more required files not found (market_clearing, accepted_bids, or submitted_profiles).")


    print("\nVisualization process complete.")

if __name__ == "__main__":
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        possible_configs = ["config_eg.yaml", "config.yaml"]
        config_file_path = None
        for config_filename in possible_configs:
            path_to_check = os.path.join(script_dir, config_filename)
            if os.path.exists(path_to_check):
                config_file_path = path_to_check
                break
        
        if config_file_path is None:
            print(f"ERROR: Could not find 'config_eg.yaml' or 'config.yaml' in the script directory: {script_dir}")
            sys.exit(1)

        config = load_config(config_file_path)
        if config is None:
            print(f"Could not load configuration from {config_file_path}. Exiting.")
            sys.exit(1)

        output_dir_from_config = config.get('simulation', {}).get('output_dir')
        if not output_dir_from_config:
            print("ERROR: 'output_dir' not found in the 'simulation' section of the config file. Exiting.")
            sys.exit(1)
            
        output_dir_path = os.path.abspath(os.path.join(script_dir, output_dir_from_config))

        print(f"--- Auto-detection successful ---")
        print(f"Config file used: {config_file_path}")
        print(f"Output directory: {output_dir_path}")
        print(f"---------------------------------")

        main(output_dir_path, config_file_path)

    except Exception as e:
        print("\n--- An error occurred during automated path setup ---")
        print(f"Error: {e}")
        traceback.print_exc()
        print("Please ensure this script is in the same directory as your config file (e.g., 'config_eg.yaml').")
        sys.exit(1)
