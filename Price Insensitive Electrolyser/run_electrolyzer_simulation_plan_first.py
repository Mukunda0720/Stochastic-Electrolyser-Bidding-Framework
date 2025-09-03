# run_electrolyzer_simulation_plan_first.py
# V1.0.0 - Entry point for the "Plan First, Buy Later" electrolyzer strategy.

"""
Main Script for Running Electrolyzer Rolling Horizon Optimization
with the deterministic "Plan First, Buy Later" strategy.

This script serves as the entry point for running the simulation.
It loads the configuration and runs the simulation using the
ElectrolyzerRollingHorizonManagerPlanFirst.

Usage:
    python run_electrolyzer_simulation_plan_first.py [--config CONFIG_FILE]
"""

import argparse
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend

# Import the new manager class for the "Plan First" strategy
from electrolyzer_rolling_horizon_plan_first import ElectrolyzerRollingHorizonManagerPlanFirst

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run Electrolyzer "Plan First" Rolling Horizon Optimization.')
    
    # Point to the new config file by default
    parser.add_argument(
        '--config', 
        default='config_plan_first.yaml', 
        help='Path to the config file for the "Plan First" strategy'
    )
    return parser.parse_args()

def main():
    """Main function to run the simulation."""
    args = parse_arguments()

    if not os.path.exists(args.config):
        print(f"FATAL ERROR: Configuration file '{args.config}' not found.")
        return 1

    try:
        start_time = datetime.now()
        print(f"Starting Electrolyzer 'Plan First, Buy Later' simulation at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Using configuration: {args.config}")

        # Instantiate the correct manager for the "Plan First" strategy
        manager = ElectrolyzerRollingHorizonManagerPlanFirst(args.config)
        
        # Run the simulation
        results = manager.run_simulation()

        end_time = datetime.now()
        elapsed = end_time - start_time
        print(f"\n'Plan First' simulation completed in {elapsed.total_seconds():.1f} seconds ({elapsed})")
        
        return 0 # Indicate success

    except Exception as e:
        print(f"ERROR: An unexpected error occurred during the simulation: {e}")
        import traceback
        traceback.print_exc()
        return 1 # Indicate failure

if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
