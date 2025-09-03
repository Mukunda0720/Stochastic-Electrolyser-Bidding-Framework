# plot_perfect_information_results.py
# V1.6 - Reverted to creating separate plots for each plant's electricity consumption.
#      - Created a single combined plot for electricity consumption of all plants (in MWh).
#      - Removed the individual plant consumption plots.
#      - Corrected file loading logic to match filenames starting with '_'.
#      - Calculates cumulative production from compressor operations.
#      - Simplified plots to match available data.

"""
Generates plots for visualizing the perfect information simulation results.

Applicable plots include:
- Separate plant electricity consumption vs. market price.
- Storage level monitoring vs. market price.
- Cumulative production tracking against the simulation target.
- Total site electricity consumption overview.

Usage (VS Code / Terminal):
    1. Place this script in the same directory as your 'config_perfect_info.yaml' file.
    2. Run the script: python plot_perfect_information_results.py
    3. It will automatically find the config and the output directory listed within it.
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
import argparse

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

def load_simulation_data(output_dir, config):
    """
    Loads all necessary output files from the perfect information simulation.
    """
    # CORRECTED: The actual output files start with an underscore.
    file_prefix = "_"
    print(f"INFO: Using file prefix: '{file_prefix}' to match the actual output files.")

    # CORRECTED: Filenames are constructed with the underscore prefix.
    files_to_load = {
        "plant_operations": f"{file_prefix}hourly_plant_operations.xlsx",
        "storage_levels": f"{file_prefix}hourly_storage_levels.xlsx",
        "site_consumption": f"{file_prefix}hourly_site_consumption.xlsx",
    }
    
    loaded_data = {}
    for key, filename in files_to_load.items():
        filepath = os.path.join(output_dir, filename)
        df = pd.DataFrame()
        if os.path.exists(filepath):
            try:
                temp_df = pd.read_excel(filepath)
                # Ensure all potential date columns are parsed correctly
                for col in temp_df.columns:
                    if 'date' in col.lower() or 'time' in col.lower():
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
    sim_conf = config.get('simulation', {})
    
    # Check if synthetic data was used first
    if config.get('synthetic_data', {}).get('enabled', False):
        # The synthetic price file uses the directory name as a prefix
        dir_prefix = os.path.basename(output_dir.rstrip(os.sep))
        synth_file = os.path.join(output_dir, f"{dir_prefix}_synthetic_actual_prices.xlsx")
        if os.path.exists(synth_file):
            try:
                market_data_full = pd.read_excel(synth_file).rename(columns={'Synthetic_Actual_MCP': 'MCP', 'Date': 'Timestamp'})
                market_data_full['Timestamp'] = pd.to_datetime(market_data_full['Timestamp']).dt.tz_localize(None)
                print(f"Successfully loaded synthetic market data from {synth_file}")
            except Exception as e:
                 print(f"Warning: Could not load synthetic market data from {synth_file}: {e}")
    else:
        market_input_file = sim_conf.get('market_input_file')
        if market_input_file and os.path.exists(market_input_file):
            try:
                market_data_full = pd.read_excel(market_input_file).rename(columns={'Price': 'MCP', 'Date': 'Timestamp'})
                market_data_full['Timestamp'] = pd.to_datetime(market_data_full['Timestamp']).dt.tz_localize(None)
                print(f"Successfully loaded full market data from {market_input_file}")
            except Exception as e:
                print(f"Warning: Could not load full market data from {market_input_file}: {e}")

    return loaded_data, market_data_full


def plot_plant_electricity_consumption(plant_name, plant_df, market_data_full, plot_output_dir):
    """
    Plots the electricity consumption for a single plant against market price.
    """
    if plant_df.empty:
        print(f"No operational data to plot for {plant_name}.")
        return

    plot_df = plant_df.copy()
    # Generalize electricity calculation
    elec_factor = abs(ELECTRICITY_INPUT_MAP.get(plant_name, 0.0))
    plot_df['Electricity_MWh'] = plot_df['Operation_Level'] * elec_factor
    
    plot_df.sort_values('Timestamp', inplace=True)
    plot_df.fillna(0, inplace=True)

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle(f"Electrical Consumption for {plant_name} (Perfect Information)", fontsize=18, y=0.98)
    
    # Subplot 1: Electricity Consumption
    ax1.set_title("Electricity Consumption (MWh)", fontsize=12)
    ax1.plot(plot_df['Timestamp'], plot_df['Electricity_MWh'],
             label=f'Electricity Consumption (MWh)', color='navy', linewidth=1.5)

    ax1.set_ylabel('Electricity (MWh)')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Subplot 2: Market Price Context
    ax2.set_title("Market Clearing Price", fontsize=12)
    if not market_data_full.empty and not plot_df.empty:
        min_ts, max_ts = plot_df['Timestamp'].min(), plot_df['Timestamp'].max()
        mcp_plot_df = market_data_full[(market_data_full['Timestamp'] >= min_ts) & (market_data_full['Timestamp'] <= max_ts)]
        ax2.plot(mcp_plot_df['Timestamp'], mcp_plot_df['MCP'], 
                 label='MCP (€/MWh)', color='purple', linewidth=1.5)
    else:
        ax2.text(0.5, 0.5, 'Market data not available.', ha='center', va='center', transform=ax2.transAxes)
        
    ax2.set_ylabel('Price (€/MWh)')
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle=':', alpha=0.6)

    ax2.set_xlabel('Timestamp')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot_filename = os.path.join(plot_output_dir, f"{plant_name.replace(' ', '_')}_consumption_perfect_info.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)


def plot_storage_level(material_name, material_df, market_data_full, plot_output_dir):
    """
    Plots the hourly storage level for a single material with a market price subplot.
    """
    if material_df.empty:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle(f"Storage and Market Analysis for {material_name} (Perfect Information)", fontsize=16, y=0.98)

    # Subplot 1: Storage Level
    ax1.set_title(f"Hourly Storage Level for {material_name}", fontsize=12)
    # Correct column name is 'Storage_Level'
    ax1.plot(material_df['Timestamp'], material_df['Storage_Level'],
             label=f'{material_name} Level (kg)', color='brown', linestyle='-', linewidth=1.5)
    
    max_data_level = material_df['Storage_Level'].max()

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
    if not market_data_full.empty and not material_df.empty:
        min_ts, max_ts = material_df['Timestamp'].min(), material_df['Timestamp'].max()
        if pd.notna(min_ts) and pd.notna(max_ts):
            mcp_plot_df = market_data_full[(market_data_full['Timestamp'] >= min_ts) & (market_data_full['Timestamp'] <= max_ts)]
            ax2.plot(mcp_plot_df['Timestamp'], mcp_plot_df['MCP'], color='purple', linewidth=1.5, label='MCP (€/MWh)')
    else:
        ax2.text(0.5, 0.5, 'Market data not available.', ha='center', va='center', transform=ax2.transAxes)

    ax2.set_ylabel('Price (€/MWh)')
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle=':', alpha=0.7)

    ax2.set_xlabel('Timestamp')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot_filename = os.path.join(plot_output_dir, f"{material_name.replace(' ', '_')}_storage_analysis_perfect_info.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)


def plot_cumulative_hydrogen_production(plant_ops_df, market_data_full, config, plot_output_dir):
    """
    Calculates and plots the cumulative production of the final product against the target.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True, 
                                   gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle('Hydrogen Production and Market Price (Perfect Information)', fontsize=16, y=0.99)

    # Subplot 1: Cumulative Production
    ax1.set_title('Cumulative Hydrogen Production', fontsize=12)
    
    sim_params = config.get('simulation', {})
    sim_start_date = pd.to_datetime(sim_params.get('start_date')).tz_localize(None)
    sim_end_date_str = sim_params.get('end_date')

    # Calculate cumulative production from Compressor operations
    if not plant_ops_df.empty:
        compressor_df = plant_ops_df[plant_ops_df['Plant_Name'] == 'Compressor'].copy()
        compressor_df.sort_values('Timestamp', inplace=True)
        compressor_df['Cumulative_H2_kg'] = compressor_df['Operation_Level'].cumsum()
        
        ax1.plot(compressor_df['Timestamp'], compressor_df['Cumulative_H2_kg'], 
                 label='Cumulative Actual Hydrogen (kg)', color='green', linestyle='-', linewidth=2)

    # Plot the overall simulation target line
    plant_params = config.get('electrolyzer_plant', {})
    target_h2_per_day = plant_params.get('target_hydrogen_per_day', 0)

    if target_h2_per_day > 0 and sim_start_date and sim_end_date_str:
        sim_end_date = pd.to_datetime(sim_end_date_str).tz_localize(None)
        # Create a target line for each day
        target_dates = pd.date_range(start=sim_start_date, end=sim_end_date, freq='D')
        cumulative_target = [((td.date() - sim_start_date.date()).days + 1) * target_h2_per_day for td in target_dates]
        ax1.plot(target_dates, cumulative_target, label='Cumulative Target Hydrogen (kg)', color='red', linestyle=':', linewidth=2.5)

    if plant_ops_df.empty and not (target_h2_per_day > 0):
         ax1.text(0.5, 0.5, 'No Hydrogen production data to plot.', ha='center', va='center', transform=ax1.transAxes)
    ax1.set_ylabel('Cumulative Hydrogen (kg)')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.7)

    # Subplot 2: Market Price
    ax2.set_title('Market Clearing Price', fontsize=12)
    if not market_data_full.empty and sim_start_date and sim_end_date_str:
        sim_end_date = pd.to_datetime(sim_end_date_str).tz_localize(None)
        mcp_plot_df = market_data_full[(market_data_full['Timestamp'] >= sim_start_date) & (market_data_full['Timestamp'] <= sim_end_date)]
        ax2.plot(mcp_plot_df['Timestamp'], mcp_plot_df['MCP'], color='purple', linewidth=1.5, label='MCP (€/MWh)')
    else:
        ax2.text(0.5, 0.5, 'Market data context not available.', ha='center', va='center', transform=ax2.transAxes)

    ax2.set_ylabel('Price (€/MWh)')
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle=':', alpha=0.7)
    
    ax2.set_xlabel('Date', fontsize=12)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    plot_filename = os.path.join(plot_output_dir, "cumulative_hydrogen_production_perfect_info.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)


def plot_electricity_overview(site_consumption_df, market_data_full, plot_output_dir):
    """Plots an overview of total hourly electricity consumption and market price."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle('Total Electricity and Market Price Overview (Perfect Information)', fontsize=16, y=0.98)
    
    if site_consumption_df.empty:
        print("No electricity data available to plot overview.")
        plt.close(fig)
        return

    # --- Subplot 1: Total Hourly Electricity ---
    ax1.set_title('Total Hourly Site Electricity Consumption', fontsize=12)
    
    ax1.plot(site_consumption_df['Timestamp'], site_consumption_df['Total_Site_Consumption_MWh'], 
             label='Total Site Consumption (MWh)', color='green', linestyle='-', alpha=0.9, linewidth=1.5)

    ax1.set_ylabel('Total Electricity (MWh)')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.7)

    # --- Subplot 2: Market Clearing Price ---
    ax2.set_title('Market Clearing Price', fontsize=12)
    if not market_data_full.empty and not site_consumption_df.empty:
        min_ts, max_ts = site_consumption_df['Timestamp'].min(), site_consumption_df['Timestamp'].max()
        if pd.notna(min_ts) and pd.notna(max_ts):
            mcp_plot_df = market_data_full[(market_data_full['Timestamp'] >= min_ts) & (market_data_full['Timestamp'] <= max_ts)]
            ax2.plot(mcp_plot_df['Timestamp'], mcp_plot_df['MCP'], color='purple', linewidth=1.5, label='MCP (€/MWh)')
    else:
        ax2.text(0.5, 0.5, 'Market data not available.', ha='center', va='center', transform=ax2.transAxes)

    ax2.set_ylabel('Price (€/MWh)')
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle=':', alpha=0.7)

    # --- Final Layout Adjustments ---
    ax2.set_xlabel('Timestamp')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot_filename = os.path.join(plot_output_dir, "total_electricity_overview_perfect_info.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)


def main():
    """Main function to drive the plotting process."""
    parser = argparse.ArgumentParser(description="Generate plots for Perfect Information simulation results.")
    parser.add_argument(
        '--config',
        default='config_perfect_info.yaml',
        help='Path to the configuration file (default: config_perfect_info.yaml)'
    )
    args = parser.parse_args()

    print(f"--- Starting Visualization for Perfect Information Model ---")
    
    config = load_config(args.config)
    if config is None:
        sys.exit(1)
        
    output_dir_from_config = config.get('simulation', {}).get('output_dir')
    if not output_dir_from_config:
        print("ERROR: 'output_dir' not found in the 'simulation' section of the config file. Exiting.")
        sys.exit(1)
        
    # Handle both relative and absolute paths in the config file
    if not os.path.isabs(output_dir_from_config):
        output_dir_path = os.path.abspath(os.path.join(os.path.dirname(args.config), output_dir_from_config))
    else:
        output_dir_path = output_dir_from_config

    print(f"Output directory: {output_dir_path}")

    loaded_data, market_data_full = load_simulation_data(output_dir_path, config)
    
    plot_main_dir = os.path.join(output_dir_path, "simulation_plots_perfect_info")
    os.makedirs(plot_main_dir, exist_ok=True)
    print(f"Plots will be saved in: {plot_main_dir}")

    # --- 1. Plot Plant Operations ---
    plant_plot_dir = os.path.join(plot_main_dir, "plant_operations")
    os.makedirs(plant_plot_dir, exist_ok=True)
    
    plant_ops_df = loaded_data.get("plant_operations", pd.DataFrame())
    
    if not plant_ops_df.empty:
        # Loop through all flexible plants and plot their electricity consumption
        for plant_name in sorted(FLEXIBLE_PLANTS):
            plant_df = plant_ops_df[plant_ops_df['Plant_Name'] == plant_name]
            plot_plant_electricity_consumption(plant_name, plant_df, market_data_full, plant_plot_dir)


    # --- 2. Plot Storage Levels ---
    storage_plot_dir = os.path.join(plot_main_dir, "storage_levels")
    os.makedirs(storage_plot_dir, exist_ok=True)
    storage_data = loaded_data.get("storage_levels", pd.DataFrame())
    if not storage_data.empty and 'Material' in storage_data.columns:
        for material in storage_data['Material'].unique():
            material_df = storage_data[storage_data['Material'] == material]
            plot_storage_level(material, material_df, market_data_full, storage_plot_dir)

    # --- 3. Plot Cumulative Production ---
    prod_summary_dir = os.path.join(plot_main_dir, "production_summary")
    os.makedirs(prod_summary_dir, exist_ok=True)
    plot_cumulative_hydrogen_production(plant_ops_df, market_data_full, config, prod_summary_dir)

    # --- 4. Plot Electricity Overview ---
    elec_dir = os.path.join(plot_main_dir, "electricity_analysis")
    os.makedirs(elec_dir, exist_ok=True)
    site_consumption_df = loaded_data.get("site_consumption", pd.DataFrame())
    plot_electricity_overview(site_consumption_df, market_data_full, elec_dir)

    print("\n--- Visualization process complete. ---")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n--- An unexpected error occurred in the main execution block ---")
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)
