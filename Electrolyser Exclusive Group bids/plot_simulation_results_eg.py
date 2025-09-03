# plot_simulation_results_eg.py
# V6 - DESCRIPTION: Removed dependency on forecast_errors.xlsx to fix crash.
# - The script now finds forecast files by scanning the output directory directly.
# - This version is fully generic for both steel and electrolyzer plants.
# - All plots use a consistent, clear style with market context.

"""
Generates plots for visualizing Exclusive Group Bid simulation results for either
a steel plant or an electrolyzer.

Includes:
- A simplified plant analysis focusing on electricity consumption and market price.
- A comprehensive profile plot showing the accepted bid, the top-ranked bid,
  and all other submitted bids against the market price.
- Actual storage levels for materials with market context and dynamic scaling.
- Price forecasts vs. actuals.
- Cumulative production (actual vs. target) with market prices.

Usage:
    Place this script in the same directory as your 'config_eg.yaml' file and run it.
    python plot_simulation_results_eg.py
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
import re

# --- ADAPTATION: Import definitions for the plant system ---
# This script can be adapted for steel or electrolyzer by changing definitions
try:
    from steel_plant_definitions import PLANT_DEFINITIONS, ABSOLUTE_PLANT_LIMITS, STORAGE_DEFINITIONS
    PLANT_TYPE = 'steel_plant'
    print("INFO: Detected STEEL PLANT definitions.")
except ImportError:
    try:
        from electrolyzer_definitions import PLANT_DEFINITIONS, ABSOLUTE_PLANT_LIMITS, STORAGE_DEFINITIONS
        PLANT_TYPE = 'electrolyzer_plant'
        print("INFO: Detected ELECTROLYZER definitions.")
    except ImportError as e:
        print(f"ERROR: Could not import necessary definition modules: {e}")
        sys.exit(1)

from ml_forecaster import ElectricityPriceForecaster

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

def load_eg_model_data(output_dir, config):
    """Loads all necessary data files from an EG simulation output directory."""
    try:
        dir_from_config = config.get('simulation',{}).get('output_dir','unknown')
        output_dir_basename = os.path.basename(dir_from_config.rstrip(os.sep))
    except Exception:
        output_dir_basename = os.path.basename(output_dir.rstrip(os.sep))

    print(f"DEBUG: Using output file prefix: '{output_dir_basename}'")

    if PLANT_TYPE == 'steel_plant':
        production_filename = f"{output_dir_basename}_hot_rolled_steel_implemented_periodical_actual_redispatch.xlsx"
    else: # Assumes electrolyzer
        production_filename = f"{output_dir_basename}_hydrogen_produced_periodical_actual.xlsx"

    files_to_try = {
        "consumption_data_actual": os.path.join(output_dir, f"{output_dir_basename}_plant_operations_details_actual_redispatch.xlsx"),
        "storage_data_hourly_actual": os.path.join(output_dir, f"{output_dir_basename}_hourly_storage_levels_details_actual_redispatch.xlsx"),
        "production_df_actual": os.path.join(output_dir, production_filename),
        "accepted_bids_df_full": os.path.join(output_dir, f"{output_dir_basename}_accepted_eg_bid_details.xlsx"),
        "submitted_profiles_df": os.path.join(output_dir, f"{output_dir_basename}_all_submitted_profiles_detailed.xlsx"),
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

    loaded_data["market_data_full"] = market_data_full
    return loaded_data

def plot_plant_data_eg(plant_name, plant_df_actual, market_data_full, opportunity_cost, plot_output_dir):
    """
    Plots a simplified, clearer operational analysis for a single plant (EG version),
    focusing on electricity consumption and market price with opportunity cost.
    """
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
        ax2.text(0.5, 0.5, 'Market price data not available.', ha='center', va='center', transform=ax2.transAxes)

    if opportunity_cost is not None:
        ax2.axhline(y=opportunity_cost, color='green', linestyle='--',
                    label=f'Electricity Opportunity Cost (€{opportunity_cost:.2f}/MWh)')

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


def plot_storage_data_eg(material_name, material_df_actual, market_data_full, opportunity_cost, plot_output_dir):
    """
    Plots storage level data with dynamic scaling and a market price subplot.
    """
    if material_df_actual.empty: return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle(f"Actual Storage and Market Analysis for {material_name} (EG)", fontsize=16, y=0.98)

    # --- Subplot 1: Storage Level ---
    ax1.set_title(f"Actual Hourly Storage Level for {material_name}", fontsize=12)
    ax1.plot(material_df_actual['Timestamp'], material_df_actual['Actual_Storage_Level_Tons'], label=f'Actual {material_name} Level (tons)', color='brown', linestyle='-', linewidth=1.5)
    
    storage_meta = STORAGE_DEFINITIONS.get(material_name, {})
    min_level, max_level = storage_meta.get('Min_Level'), storage_meta.get('Max_Level')
    if pd.notna(min_level): ax1.axhline(y=min_level, color='red', linestyle=':', label=f'Min Level ({min_level:,.0f} tons)')
    if pd.notna(max_level) and max_level < 1e8: ax1.axhline(y=max_level, color='green', linestyle=':', label=f'Max Level ({max_level:,.0f} tons)')

    # Dynamic Y-axis scaling
    max_data_level = material_df_actual['Actual_Storage_Level_Tons'].max()
    if max_data_level > 0:
        y_upper_limit = max_data_level * 1.15
        y_lower_limit = -0.05 * y_upper_limit
        ax1.set_ylim(bottom=y_lower_limit, top=y_upper_limit)
    else:
        ax1.set_ylim(bottom=-1, top=10)

    ax1.set_ylabel(f'{material_name} Level (tons)'); ax1.legend(loc='upper left'); ax1.grid(True, linestyle=':', alpha=0.7)

    # --- Subplot 2: Market Price ---
    ax2.set_title("Actual Market Clearing Price & Opportunity Cost", fontsize=12)
    if not market_data_full.empty:
        min_ts, max_ts = material_df_actual['Timestamp'].min(), material_df_actual['Timestamp'].max()
        if pd.notna(min_ts) and pd.notna(max_ts):
            mcp_plot_df = market_data_full[(market_data_full['Timestamp'] >= min_ts) & (market_data_full['Timestamp'] <= max_ts)]
            ax2.plot(mcp_plot_df['Timestamp'], mcp_plot_df['MCP'], label='MCP (€/MWh)', color='purple', linewidth=1.5)
    else:
        ax2.text(0.5, 0.5, 'Market price data not available.', ha='center', va='center', transform=ax2.transAxes)

    if opportunity_cost is not None:
        ax2.axhline(y=opportunity_cost, color='green', linestyle='--', label=f'Electricity Opportunity Cost (€{opportunity_cost:.2f}/MWh)')

    ax2.set_ylabel('Price (€/MWh)'); ax2.legend(loc='upper left'); ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.set_xlabel('Timestamp')
    
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(os.path.join(plot_output_dir, f"{material_name.replace(' ', '_')}_storage_level_analysis_eg.png"), dpi=300)
    plt.close(fig)

def plot_cumulative_production_eg(production_df_actual, market_data_full, opportunity_cost, config, plot_output_dir):
    """
    Plots cumulative production and market prices in two separate, properly scaled subplots.
    This function is generic for either steel or hydrogen.
    """
    if production_df_actual.empty:
        print("Warning: Production data is empty. Cannot plot cumulative production.")
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.text(0.5, 0.5, 'Production data not available.', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Cumulative Production & Market Price (EG)', fontsize=16)
        plt.savefig(os.path.join(plot_output_dir, "cumulative_production_analysis_eg.png"), dpi=300)
        plt.close(fig)
        return

    # --- Determine if this is a steel or hydrogen plant based on columns/config ---
    if 'total_actual_hot_rolled_steel_tons' in production_df_actual.columns:
        prod_col = 'total_actual_hot_rolled_steel_tons'
        target_col = 'target_hot_rolled_steel_per_day'
        unit = 'tons'
        product_name = 'Hot Rolled Steel'
    elif 'total_actual_hydrogen_kg' in production_df_actual.columns:
        prod_col = 'total_actual_hydrogen_kg'
        target_col = 'target_hydrogen_per_day'
        unit = 'kg'
        product_name = 'Hydrogen'
    else:
        print("Could not determine production type (steel or hydrogen). Skipping plot.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle(f'{product_name} Production and Market Price Analysis (EG)', fontsize=16, y=0.99)

    # --- Subplot 1: Cumulative Production ---
    ax1.set_title(f'Cumulative {product_name} Production', fontsize=12)
    
    sort_col = 'end_date' if 'end_date' in production_df_actual.columns else 'start_date'
    actual_prod_plot_df = production_df_actual.sort_values(by=sort_col).copy()
    actual_prod_plot_df[f'cumulative_actual_{unit}'] = actual_prod_plot_df[prod_col].cumsum()
    
    ax1.plot(actual_prod_plot_df[sort_col], actual_prod_plot_df[f'cumulative_actual_{unit}'], 
             label=f'Cumulative Actual {product_name} ({unit})', color='green', marker='x', linestyle='-', markersize=5, linewidth=2)

    sim_params = config.get('simulation', {})
    plant_params = config.get(PLANT_TYPE, {})
    sim_start_date = pd.to_datetime(sim_params.get('start_date')).tz_localize(None)
    sim_end_date = pd.to_datetime(sim_params.get('end_date')).tz_localize(None)
    target_per_day = plant_params.get(target_col, 0)

    if target_per_day > 0 and pd.notna(sim_start_date) and pd.notna(sim_end_date):
        target_dates = pd.date_range(start=sim_start_date, end=sim_end_date, freq='D')
        cumulative_target = [((d.date() - sim_start_date.date()).days + 1) * target_per_day for d in target_dates]
        ax1.plot(target_dates, cumulative_target, label=f'Cumulative Target {product_name} ({unit})', color='red', linestyle=':', linewidth=2.5)

    ax1.set_ylabel(f'Cumulative {product_name} ({unit})')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.7)

    # --- Subplot 2: Market Clearing Price & Opportunity Cost ---
    ax2.set_title('Market Clearing Price & Opportunity Cost', fontsize=12)
    if not market_data_full.empty:
        min_ts, max_ts = actual_prod_plot_df[sort_col].min(), actual_prod_plot_df[sort_col].max()
        if pd.notna(min_ts) and pd.notna(max_ts):
            mcp_plot_df = market_data_full[(market_data_full['Timestamp'] >= min_ts) & (market_data_full['Timestamp'] <= max_ts)]
            ax2.plot(mcp_plot_df['Timestamp'], mcp_plot_df['MCP'], color='purple', linewidth=1.5, label='Actual MCP (€/MWh)')
    else:
        ax2.text(0.5, 0.5, 'Complete market data not available.', ha='center', va='center', transform=ax2.transAxes)

    if opportunity_cost is not None:
        ax2.axhline(y=opportunity_cost, color='green', linestyle='--', label=f'Electricity Opportunity Cost (€{opportunity_cost:.2f}/MWh)')

    ax2.set_ylabel('Price (€/MWh)')
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle=':', alpha=0.7)
    
    ax2.set_xlabel('Date', fontsize=12)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    plt.savefig(os.path.join(plot_output_dir, "cumulative_production_analysis_eg.png"), dpi=300)
    plt.close(fig)

def plot_total_electricity_overview_eg(consumption_data_actual, accepted_bids_df_full, market_data_full, opportunity_cost, config, plot_output_dir):
    """Plots total electricity consumption with a market price subplot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle('Total Electricity and Market Price Overview (EG)', fontsize=16, y=0.98)

    baseload_mw = config.get(PLANT_TYPE, {}).get('baseload_mw', 0.0)

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
        ax1.text(0.5, 0.5, 'No electricity data.', ha='center', va='center', transform=ax1.transAxes)
        plt.close(fig); return

    plot_df = pd.DataFrame(sorted(list(all_timestamps)), columns=['Timestamp'])
    if not actual_consumed_flex.empty: plot_df = pd.merge(plot_df, actual_consumed_flex, on='Timestamp', how='left')
    if not actual_cleared_total.empty: plot_df = pd.merge(plot_df, actual_cleared_total, on='Timestamp', how='left')
    plot_df.fillna(0, inplace=True)

    # --- Subplot 1: Electricity Overview ---
    ax1.set_title("Total Hourly Electricity Overview", fontsize=12)
    if 'Flex_Plants_Consumed_MWh' in plot_df.columns:
        plot_df['Total_Actual_Consumed_MWh'] = plot_df['Flex_Plants_Consumed_MWh'] + baseload_mw
        ax1.plot(plot_df['Timestamp'], plot_df['Total_Actual_Consumed_MWh'], label='Total Actual Consumption', color='green', linestyle='-', alpha=0.9, linewidth=2)

    if 'Total_Cleared_MWh' in plot_df.columns:
        ax1.plot(plot_df['Timestamp'], plot_df['Total_Cleared_MWh'], label='Total Cleared from Market', color='black', marker='.', markersize=4, alpha=0.9, linewidth=1.5)

    ax1.set_ylabel('Total Electricity (MWh)')
    ax1.legend(loc='upper left'); ax1.grid(True, linestyle=':', alpha=0.7)

    # --- Subplot 2: Market Price ---
    ax2.set_title("Actual Market Clearing Price & Opportunity Cost", fontsize=12)
    if not market_data_full.empty:
        min_ts, max_ts = plot_df['Timestamp'].min(), plot_df['Timestamp'].max()
        if pd.notna(min_ts) and pd.notna(max_ts):
            mcp_plot_df = market_data_full[(market_data_full['Timestamp'] >= min_ts) & (market_data_full['Timestamp'] <= max_ts)]
            ax2.plot(mcp_plot_df['Timestamp'], mcp_plot_df['MCP'], label='MCP (€/MWh)', color='purple', linewidth=1.5)
    else:
        ax2.text(0.5, 0.5, 'Market price data not available.', ha='center', va='center', transform=ax2.transAxes)

    if opportunity_cost is not None:
        ax2.axhline(y=opportunity_cost, color='green', linestyle='--', label=f'Electricity Opportunity Cost (€{opportunity_cost:.2f}/MWh)')

    ax2.set_ylabel('Price (€/MWh)'); ax2.legend(loc='upper left'); ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.set_xlabel('Timestamp')
    
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45, ha='right'); plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(os.path.join(plot_output_dir, "total_electricity_overview_eg.png"), dpi=300); plt.close(fig)

def plot_forecast_vs_actual_window(period, window_start_date, forecast_file_path, config, market_data_full, plot_output_dir):
    """Plots forecast vs actual prices for a specific simulation window, with no metrics box."""
    try:
        forecast_df = pd.read_csv(forecast_file_path)
        forecast_df['Date'] = pd.to_datetime(forecast_df['Date']).dt.tz_localize(None)

        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(forecast_df['Date'], forecast_df['Forecast_Price'], label='Forecast', color='blue', linewidth=1.5)
        
        if not market_data_full.empty:
            actual_df_window = market_data_full[market_data_full['Timestamp'].isin(forecast_df['Date'])]
            ax.plot(actual_df_window['Timestamp'], actual_df_window['MCP'], label='Actual', color='red', linestyle='--', alpha=0.7)

        ax.set_title(f'Period {period}: Price Forecast vs Actual'); ax.set_xlabel('Date'); ax.set_ylabel('Price (€/MWh)')
        ax.legend(); ax.grid(True, alpha=0.5); fig.autofmt_xdate(); plt.tight_layout()
        plt.savefig(os.path.join(plot_output_dir, f"period_{period}_forecast_vs_actual_eg.png"), dpi=300); plt.close()
    except Exception as e:
        print(f"ERROR plotting forecast for period {period} from file {forecast_file_path}: {e}")

def plot_profile_analysis_with_context(period_info, accepted_bids_df, submitted_profiles_df, config, plot_output_dir):
    """
    Plots the verified accepted profile, highlights the top-ranked profile, and shows all other
    submitted profiles as context.
    """
    if period_info is None or pd.isna(period_info.get('period')):
        print(f"Warning: Skipping profile analysis plot due to invalid period information in row: {period_info}")
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


def main():
    """Main function to orchestrate loading data and generating all plots."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_file_path = os.path.join(script_dir, "config_eg.yaml")
        
        config = load_config(config_file_path)
        if config is None:
            sys.exit(1)

        output_dir_arg = config.get('simulation', {}).get('output_dir')
        if not output_dir_arg:
            print("FATAL: 'output_dir' not specified in the simulation config.")
            sys.exit(1)
        
        output_dir_path = os.path.abspath(os.path.join(script_dir, output_dir_arg))
        print(f"--- Auto-detection successful ---")
        print(f"Config file: {config_file_path}")
        print(f"Output directory: {output_dir_path}")
        print(f"---------------------------------")
    except Exception as e:
        print(f"Error during auto-detection: {e}")
        sys.exit(1)

    print(f"Starting EG visualization process for {PLANT_TYPE}: {output_dir_path}")

    loaded_data = load_eg_model_data(output_dir_path, config)
    plot_main_dir = os.path.join(output_dir_path, f"simulation_plots_{PLANT_TYPE}_eg")
    os.makedirs(plot_main_dir, exist_ok=True)

    plot_dirs = {
        "plant_ops": os.path.join(plot_main_dir, "plant_operations"),
        "storage": os.path.join(plot_main_dir, "storage_levels"),
        "production": os.path.join(plot_main_dir, "production_summary"),
        "electricity": os.path.join(plot_main_dir, "electricity_overview"),
        "forecasts": os.path.join(plot_main_dir, "forecast_evaluations"),
        "profiles": os.path.join(plot_main_dir, "profile_selection_analysis")
    }
    for d in plot_dirs.values(): os.makedirs(d, exist_ok=True)

    opportunity_cost = config.get(PLANT_TYPE, {}).get('opportunity_cost_mwh_shortfall')
    if opportunity_cost is not None:
        print(f"INFO: Found opportunity_cost_mwh_shortfall in config: {opportunity_cost}")
    else:
        print(f"WARNING: 'opportunity_cost_mwh_shortfall' not found under '{PLANT_TYPE}' in the config file. The line will not be plotted.")

    if not loaded_data['consumption_data_actual'].empty:
        for name in loaded_data['consumption_data_actual']['Plant_Name'].unique():
            df = loaded_data['consumption_data_actual'][loaded_data['consumption_data_actual']['Plant_Name'] == name]
            plot_plant_data_eg(name, df, loaded_data['market_data_full'], opportunity_cost, plot_dirs["plant_ops"])

    if not loaded_data['storage_data_hourly_actual'].empty:
        for name in loaded_data['storage_data_hourly_actual']['Material'].unique():
            df = loaded_data['storage_data_hourly_actual'][loaded_data['storage_data_hourly_actual']['Material'] == name]
            plot_storage_data_eg(name, df, loaded_data['market_data_full'], opportunity_cost, plot_dirs["storage"])

    plot_cumulative_production_eg(
        loaded_data['production_df_actual'],
        loaded_data['market_data_full'],
        opportunity_cost,
        config,
        plot_dirs["production"]
    )

    plot_total_electricity_overview_eg(
        loaded_data['consumption_data_actual'], 
        loaded_data['accepted_bids_df_full'], 
        loaded_data['market_data_full'],
        opportunity_cost,
        config, 
        plot_dirs["electricity"]
    )

    # --- MODIFICATION: Find and loop through forecast files directly ---
    forecast_plot_dir = plot_dirs["forecasts"]
    forecast_load_dir = os.path.join(output_dir_path, 'forecast_outputs_per_window_eg')

    if os.path.exists(forecast_load_dir):
        forecast_files = glob.glob(os.path.join(forecast_load_dir, "forecast_eg_p*.csv"))
        
        for f_path in forecast_files:
            basename = os.path.basename(f_path)
            match = re.search(r"forecast_eg_p(\d+)_(\d{8})_h", basename)
            if match:
                period = int(match.group(1))
                date_str = match.group(2)
                window_start_date = pd.to_datetime(date_str, format='%Y%m%d')
                
                plot_forecast_vs_actual_window(
                    period, 
                    window_start_date,
                    f_path,
                    config, 
                    loaded_data['market_data_full'], 
                    forecast_plot_dir
                )
            else:
                print(f"Warning: Could not parse forecast filename: {basename}")
    else:
        print("Warning: Forecast output directory not found. Skipping forecast plots.")


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
    main()
