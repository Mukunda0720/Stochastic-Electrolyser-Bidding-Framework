# run_electrolyzer_simulation_eg.py exclusive group bids rolling horizon manager    

"""
Main Script for Running Electrolyzer Rolling Horizon Optimization
with Exclusive Group Bids (EGs).
"""

import argparse
import yaml
import os
from datetime import datetime
# Import the correct manager class for EG
from electrolyzer_rolling_horizon_eg import ElectrolyzerRollingHorizonManagerEG
import matplotlib
matplotlib.use('Agg') 

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run Electrolyzer Rolling Horizon Optimization with Exclusive Group Bids.')
    parser.add_argument('--config', default='config_eg.yaml', help='Path to the EG configuration file (e.g., config_eg.yaml)')
    return parser.parse_args()

def main():
    """Main function to run the electrolyzer rolling horizon simulation with EGs."""
    args = parse_arguments()

    if not os.path.exists(args.config):
        print(f"FATAL ERROR (EG RUN): Configuration file '{args.config}' not found.")
        return 1 # Indicate failure

    try:
        start_time_sim = datetime.now()
        print(f"Starting Electrolyzer EG rolling horizon optimization at {start_time_sim.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Using configuration file: {args.config}")

        # Instantiate the EG manager
        manager = ElectrolyzerRollingHorizonManagerEG(args.config)
        results = manager.run_simulation() # Run the simulation

        end_time_sim = datetime.now()
        elapsed_sim = end_time_sim - start_time_sim
        print(f"\nElectrolyzer EG rolling horizon simulation completed in {elapsed_sim.total_seconds():.1f} seconds ({elapsed_sim})")

        return 0 # Indicate success

    except Exception as e:
        print(f"ERROR (EG RUN): An unexpected error occurred during the electrolyzer EG simulation: {e}")
        import traceback
        traceback.print_exc()
        return 1 # Indicate failure

if __name__ == '__main__':
    exit_code = main()
    if exit_code == 0:
        print("run_electrolyzer_simulation_eg.py finished successfully.")
    else:
        print("run_electrolyzer_simulation_eg.py encountered an error.")
    exit(exit_code)
