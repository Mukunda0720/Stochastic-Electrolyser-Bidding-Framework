# plot_electrolyzer_plan_first_results.py
# V1.6 - Fixed date range bug in cumulative production plot to correctly show only the simulation period.
#      - Added the Total Electricity Overview plot.
#      - Simplified plots to remove "plan vs. actual" comparison.
#      - Corrected file loading logic to match the actual '_' prefix.

"""
Generates plots for visualizing the deterministic electrolyzer simulation results.

Applicable plots include:
- Plant electricity consumption.
- Storage level monitoring.
- Cumulative production tracking against targets.
- Total electricity overview.

Usage (VS Code / Terminal):
    1. Place this script in the same directory as your 'config_plan_first.yaml' file.
    2. Run the script. It will automatically find the config and the output directory listed within it.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys
import yaml
from datetime import timedelta
import traceback
import numpy as np

# It's good practice to wrap imports that might fail
try:
    from electrolyzer_definitions import STORAGE_DEFINITIONS, ELECTRICITY_INPUT_MAP, FLEXIBLE_PLANTS
except ImportError as e:
    print(f"ERROR: Could not import from 'electrolyzer_definitions'. Please ensure the file is accessible.")
    # Provide a fallback if the script is run in a different environment
    STORAGE_DEFINITIONS, ELECTRICITY_INPUT_MAP, FLEXIBLE_PLANTS = {}, {}, []


def load_config(config_file_path):
    """Loads the YAML configuration file."""
    if not os.path.exists(config_file_path):
        print(f"ERROR: Configuration file not found: {config_file_path}")
        return None
    try:
        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Successfully loaded configuration from {config_file_path}")
        return config
    except Exception as e:
        print(f"Error loading or parsing config file {config_file_path}: {e}")
        return None

def load_deterministic_data(output_dir, config):
    """
    Loads all necessary output files from the deterministic simulation.
    """
    file_prefix = "_"
    print(f"INFO: Using file prefix: '{file_prefix}' based on observed output.")

    files_to_load = {
        "plant_operations_actual": f"{file_prefix}plant_operations_details_actual.xlsx",
        "storage_levels_actual": f"{file_prefix}hourly_storage_levels_actual.xlsx",
        "hydrogen_prod_actual": f"{file_prefix}final_product_produced_periodical_actual.xlsx",
        "accepted_bids": f"{file_prefix}accepted.xlsx",
        "site_consumption_actual": f"{file_prefix}hourly_site_consumption_actual.xlsx",
        "hourly_energy_plan": f"{file_prefix}hourly_energy_plan.xlsx",
    }
    
    loaded_data = {}
    for key, filename in files_to_load.items():
        filepath = os.path.join(output_dir, filename)
        df = pd.DataFrame()
        if os.path.exists(filepath):
            try:
                temp_df = pd.read_excel(filepath)
                for col in ['Timestamp', 'Date', 'start_date', 'end_date']:
                    if col in temp_df.columns:
                        temp_df[col] = pd.to_datetime(temp_df[col], errors='coerce').dt.tz_localize(None)
                df = temp_df
                print(f"  Successfully loaded {key} from {filename}")
            except Exception as e:
                print(f"  Warning: Error loading or processing {key} from {filepath}: {e}")
        else:
            print(f"  Warning: File not found for {key}: {filepath}")
        loaded_data[key] = df

    # Load the full market data for plotting the price context
    market_data_full = pd.DataFrame()
    market_input_file = config.get('simulation', {}).get('market_input_file')
    if market_input_file and os.path.exists(market_input_file):
        try:
            market_data_full = pd.read_excel(market_input_file).rename(columns={'Price': 'MCP', 'Date': 'Timestamp'})
            market_data_full['Timestamp'] = pd.to_datetime(market_data_full['Timestamp']).dt.tz_localize(None)
            print(f"Successfully loaded full market data from {market_input_file}")
        except Exception as e:
            print(f"Warning: Could not load full market data from {market_input_file}: {e}")

    return loaded_data, market_data_full


def plot_plant_data_deterministic(plant_name, plant_df_actual, market_data_full, plot_output_dir):
    """
    Plots the electricity consumption for a single plant.
    Simplified to show only the single "actual" plan.
    """
    if plant_df_actual.empty:
        print(f"No operational data to plot for {plant_name}.")
        return

    # Calculate actual electricity consumption for the specific plant
    elec_factor = abs(ELECTRICITY_INPUT_MAP.get(plant_name, 0.0))
    
    # Use .copy() to avoid SettingWithCopyWarning
    plot_df = plant_df_actual.copy()
    plot_df['Actual_Absolute_Electricity_MWh'] = plot_df['Actual_Operation_Level'] * elec_factor
    
    plot_df.sort_values('Timestamp', inplace=True)
    plot_df.fillna(0, inplace=True)

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle(f"Electrical Consumption for {plant_name}", fontsize=18, y=0.98)
    
    # Subplot 1: Electricity Consumption
    ax1.set_title("Electricity Consumption (MWh)", fontsize=12)
    ax1.plot(plot_df['Timestamp'], plot_df['Actual_Absolute_Electricity_MWh'],
             label=f'Electricity Consumption (MWh)', color='navy', linewidth=1.5)

    ax1.set_ylabel('Electricity (MWh)')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Subplot 2: Market Price Context
    ax2.set_title("Actual Market Clearing Price", fontsize=12)
    if not market_data_full.empty and not plot_df.empty:
        min_ts, max_ts = plot_df['Timestamp'].min(), plot_df['Timestamp'].max()
        mcp_plot_df = market_data_full[(market_data_full['Timestamp'] >= min_ts) & (market_data_full['Timestamp'] <= max_ts)]
        ax2.plot(mcp_plot_df['Timestamp'], mcp_plot_df['MCP'], 
                 label='MCP (€/MWh)', color='purple', linewidth=1.5)
    else:
        ax2.text(0.5, 0.5, 'Complete market data not available.', ha='center', va='center', transform=ax2.transAxes)
        
    ax2.set_ylabel('Price (€/MWh)')
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle=':', alpha=0.6)

    ax2.set_xlabel('Timestamp')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot_filename = os.path.join(plot_output_dir, f"{plant_name.replace(' ', '_')}_consumption_deterministic.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)


def plot_storage_data_deterministic(material_name, material_df_actual, market_data_full, plot_output_dir):
    """
    Plots the hourly storage level for a single material with a market price subplot.
    """
    if material_df_actual.empty:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle(f"Storage and Market Analysis for {material_name} (Deterministic Plan)", fontsize=16, y=0.98)

    # Subplot 1: Storage Level
    ax1.set_title(f"Hourly Storage Level for {material_name}", fontsize=12)
    ax1.plot(material_df_actual['Timestamp'], material_df_actual['Actual_Storage_Level'],
             label=f'Actual {material_name} Level (kg)', color='brown', linestyle='-', linewidth=1.5)
    
    max_data_level = material_df_actual['Actual_Storage_Level'].max()

    storage_meta = STORAGE_DEFINITIONS.get(material_name, {})
    min_level = storage_meta.get('Min_Level')
    max_level = storage_meta.get('Max_Level')
    if pd.notna(min_level):
        ax1.axhline(y=min_level, color='red', linestyle=':', label=f'Min Level ({min_level:,.0f} kg)', linewidth=1)
    if pd.notna(max_level) and max_level < 1e8:
        ax1.axhline(y=max_level, color='green', linestyle=':', label=f'Max Level ({max_level:,.0f} kg)', linewidth=1)

    # Dynamic Y-axis scaling
    if max_data_level > 0:
        y_upper_limit = max_data_level * 1.15
        y_lower_limit = -0.05 * y_upper_limit
        ax1.set_ylim(bottom=y_lower_limit, top=y_upper_limit)
    else:
        ax1.set_ylim(bottom=-1, top=10)

    ax1.set_ylabel(f'{material_name} Level (kg)')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.7)

    # Subplot 2: Market Clearing Price
    ax2.set_title('Market Clearing Price', fontsize=12)
    if not market_data_full.empty and not material_df_actual.empty:
        min_ts, max_ts = material_df_actual['Timestamp'].min(), material_df_actual['Timestamp'].max()
        if pd.notna(min_ts) and pd.notna(max_ts):
            mcp_plot_df = market_data_full[(market_data_full['Timestamp'] >= min_ts) & (market_data_full['Timestamp'] <= max_ts)]
            ax2.plot(mcp_plot_df['Timestamp'], mcp_plot_df['MCP'], color='purple', linewidth=1.5, label='Actual MCP (€/MWh)')
    else:
        ax2.text(0.5, 0.5, 'Complete market data not available.', ha='center', va='center', transform=ax2.transAxes)

    ax2.set_ylabel('Price (€/MWh)')
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle=':', alpha=0.7)

    ax2.set_xlabel('Timestamp')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot_filename = os.path.join(plot_output_dir, f"{material_name.replace(' ', '_')}_storage_analysis_deterministic.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)


def plot_cumulative_hydrogen_production_deterministic(h2_df_actual, market_data_full, config, plot_output_dir):
    """
    Plots the cumulative production of the final product against the target.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True, 
                                   gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle('Hydrogen Production and Market Price Analysis (Deterministic Plan)', fontsize=16, y=0.99)

    # Subplot 1: Cumulative Production
    ax1.set_title('Cumulative Hydrogen Production', fontsize=12)
    
    df_act = pd.DataFrame()
    sim_params = config.get('simulation', {})
    sim_start_date = pd.to_datetime(sim_params.get('start_date')).tz_localize(None)

    if not h2_df_actual.empty:
        impl_period_days = sim_params.get('implementation_period', 1)
        df_act = h2_df_actual.sort_values(by='period').copy()
        df_act['end_date'] = [sim_start_date + timedelta(days=p * impl_period_days - 1) for p in df_act['period']]
        df_act['cumulative'] = df_act['total_actual_final_product_kg'].cumsum()
        
        ax1.plot(df_act['end_date'], df_act['cumulative'], label='Cumulative Actual Hydrogen (kg)', color='green', marker='x', linestyle='-', markersize=5, linewidth=2)

    # Plot the overall simulation target line
    plant_params = config.get('electrolyzer_plant', {})
    sim_end_date_str = sim_params.get('end_date')
    target_h2_per_day = plant_params.get('target_hydrogen_per_day', 0)

    if target_h2_per_day > 0 and sim_start_date and sim_end_date_str:
        sim_end_date = pd.to_datetime(sim_end_date_str).tz_localize(None)
        target_dates = pd.date_range(start=sim_start_date, end=sim_end_date, freq='D')
        cumulative_target = [((td.date() - sim_start_date.date()).days + 1) * target_h2_per_day for td in target_dates]
        ax1.plot(target_dates, cumulative_target, label='Cumulative Target Hydrogen (kg)', color='red', linestyle=':', linewidth=2.5)

    if df_act.empty and not (target_h2_per_day > 0):
         ax1.text(0.5, 0.5, 'No Hydrogen production data to plot.', ha='center', va='center', transform=ax1.transAxes)
    ax1.set_ylabel('Cumulative Hydrogen (kg)')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.7)

    # Subplot 2: Market Price
    ax2.set_title('Market Clearing Price', fontsize=12)
    # *** FIX: Use the precise simulation start and end dates for the x-axis range ***
    if not market_data_full.empty and sim_start_date and sim_end_date_str:
        sim_end_date = pd.to_datetime(sim_end_date_str).tz_localize(None)
        mcp_plot_df = market_data_full[(market_data_full['Timestamp'] >= sim_start_date) & (market_data_full['Timestamp'] <= sim_end_date)]
        ax2.plot(mcp_plot_df['Timestamp'], mcp_plot_df['MCP'], color='purple', linewidth=1.5, label='Actual MCP (€/MWh)')
    else:
        ax2.text(0.5, 0.5, 'Market data context not available.', ha='center', va='center', transform=ax2.transAxes)

    ax2.set_ylabel('Price (€/MWh)')
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle=':', alpha=0.7)
    
    ax2.set_xlabel('Date', fontsize=12)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    plot_filename = os.path.join(plot_output_dir, "cumulative_hydrogen_production_deterministic.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)


def plot_electricity_overview_deterministic(energy_plan_df, accepted_bids_df, site_consumption_df, market_data_full, plot_output_dir):
    """Plots an overview of total hourly electricity consumption components and market price."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle('Total Electricity and Market Price Overview (Deterministic Plan)', fontsize=16, y=0.98)
    
    if accepted_bids_df.empty and site_consumption_df.empty and energy_plan_df.empty:
        print("No electricity data available to plot overview.")
        plt.close(fig)
        return

    # --- Subplot 1: Total Hourly Electricity ---
    ax1.set_title('Total Hourly Electricity Consumption', fontsize=12)
    
    # Plot 1: Cleared from Market
    if not accepted_bids_df.empty:
        cleared_df = accepted_bids_df.groupby('Date')['Accepted MW'].sum().reset_index()
        ax1.plot(cleared_df['Date'], cleared_df['Accepted MW'], label='Total Cleared from Market (MW)', color='black', marker='.', markersize=3, alpha=0.9, linewidth=1.5)

    # Plot 2: Actual Consumption
    if not site_consumption_df.empty:
        ax1.plot(site_consumption_df['Timestamp'], site_consumption_df['Total_Site_Consumption_MWh'], label='Total Actual Consumption (MWh)', color='green', linestyle='-', alpha=0.9, linewidth=1.5)

    # Plot 3: Planned Consumption
    if not energy_plan_df.empty:
        ax1.plot(energy_plan_df['Timestamp'], energy_plan_df['Planned_Energy_MWh'], label='Total Planned Consumption (MWh)', color='dodgerblue', linestyle='--', alpha=0.8, linewidth=1.5)

    ax1.set_ylabel('Total Electricity (MW/MWh)')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.7)

    # --- Subplot 2: Market Clearing Price ---
    ax2.set_title('Market Clearing Price', fontsize=12)
    plot_timestamps = pd.concat([
        df['Timestamp'] for df in [site_consumption_df, energy_plan_df] if not df.empty and 'Timestamp' in df.columns
    ] + [
        df['Date'] for df in [accepted_bids_df] if not df.empty and 'Date' in df.columns
    ])

    if not market_data_full.empty and not plot_timestamps.empty:
        min_ts, max_ts = plot_timestamps.min(), plot_timestamps.max()
        if pd.notna(min_ts) and pd.notna(max_ts):
            mcp_plot_df = market_data_full[(market_data_full['Timestamp'] >= min_ts) & (market_data_full['Timestamp'] <= max_ts)]
            ax2.plot(mcp_plot_df['Timestamp'], mcp_plot_df['MCP'], color='purple', linewidth=1.5, label='Actual MCP (€/MWh)')
    else:
        ax2.text(0.5, 0.5, 'Complete market data not available.', ha='center', va='center', transform=ax2.transAxes)

    ax2.set_ylabel('Price (€/MWh)')
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle=':', alpha=0.7)

    # --- Final Layout Adjustments ---
    ax2.set_xlabel('Timestamp')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot_filename = os.path.join(plot_output_dir, "total_electricity_overview_deterministic.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)


def main(output_dir_arg, config_file_arg):
    """Main function to drive the plotting process."""
    print(f"--- Starting Visualization for Deterministic 'Plan-First' Model ---")
    print(f"Output directory: {output_dir_arg}")
    config = load_config(config_file_arg)
    if config is None:
        sys.exit(1)

    loaded_data, market_data_full = load_deterministic_data(output_dir_arg, config)
    
    plot_main_dir = os.path.join(output_dir_arg, "simulation_plots_plan_first")
    os.makedirs(plot_main_dir, exist_ok=True)
    print(f"Plots will be saved in: {plot_main_dir}")

    # --- 1. Plot Plant Operations ---
    plant_plot_dir = os.path.join(plot_main_dir, "plant_operations")
    os.makedirs(plant_plot_dir, exist_ok=True)
    
    plant_ops_actual = loaded_data.get("plant_operations_actual", pd.DataFrame())
    
    if not plant_ops_actual.empty:
        for plant in sorted(FLEXIBLE_PLANTS):
            plant_df_act = plant_ops_actual[plant_ops_actual['Plant_Name'] == plant]
            plot_plant_data_deterministic(plant, plant_df_act, market_data_full, plant_plot_dir)

    # --- 2. Plot Storage Levels ---
    storage_plot_dir = os.path.join(plot_main_dir, "storage_levels")
    os.makedirs(storage_plot_dir, exist_ok=True)
    storage_data_act = loaded_data.get("storage_levels_actual", pd.DataFrame())
    if not storage_data_act.empty and 'Material' in storage_data_act.columns:
        for material in storage_data_act['Material'].unique():
            material_df_act = storage_data_act[storage_data_act['Material'] == material]
            plot_storage_data_deterministic(material, material_df_act, market_data_full, storage_plot_dir)

    # --- 3. Plot Cumulative Production ---
    prod_summary_dir = os.path.join(plot_main_dir, "production_summary")
    os.makedirs(prod_summary_dir, exist_ok=True)
    h2_df_act = loaded_data.get("hydrogen_prod_actual", pd.DataFrame())
    plot_cumulative_hydrogen_production_deterministic(h2_df_act, market_data_full, config, prod_summary_dir)

    # --- 4. Plot Electricity Overview ---
    elec_dir = os.path.join(plot_main_dir, "electricity_analysis")
    os.makedirs(elec_dir, exist_ok=True)
    
    energy_plan_df = loaded_data.get("hourly_energy_plan", pd.DataFrame())
    accepted_bids_df = loaded_data.get("accepted_bids", pd.DataFrame())
    site_consumption_df = loaded_data.get("site_consumption_actual", pd.DataFrame())
    
    plot_electricity_overview_deterministic(energy_plan_df, accepted_bids_df, site_consumption_df, market_data_full, elec_dir)

    print("\n--- Visualization process complete. ---")


if __name__ == "__main__":
    try:
        # Automatically find the config file and output directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_filename = "config_plan_first.yaml"
        config_file_path = os.path.join(script_dir, config_filename)

        config = load_config(config_file_path)
        if config is None:
            sys.exit(1)

        output_dir_from_config = config.get('simulation', {}).get('output_dir')
        if not output_dir_from_config:
            print("ERROR: 'output_dir' not found in the 'simulation' section of config.yaml. Exiting.")
            sys.exit(1)
            
        # Handle both relative and absolute paths in the config file
        if os.path.isabs(output_dir_from_config):
            output_dir_path = output_dir_from_config
        else:
            output_dir_path = os.path.abspath(os.path.join(script_dir, output_dir_from_config))

        main(output_dir_path, config_file_path)

    except Exception as e:
        print(f"\n--- An unexpected error occurred in the main execution block ---")
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)
