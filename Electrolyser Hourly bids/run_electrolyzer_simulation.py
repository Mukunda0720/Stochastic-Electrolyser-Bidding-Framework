# run_electrolyzer_simulation.py
# Entry point for the simplified electrolyzer model. Run this to run the model as a whole

"""
Main Script for Running Electrolyzer Rolling Horizon Optimization

Usage:
    python run_electrolyzer_simulation.py [--config CONFIG_FILE]
"""

import argparse
import os
from datetime import datetime
from electrolyzer_rolling_horizon import ElectrolyzerRollingHorizonManager
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run Electrolyzer Rolling Horizon Optimization.')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    return parser.parse_args()

def main():
    """Main function to run the electrolyzer rolling horizon simulation."""
    args = parse_arguments()

    if not os.path.exists(args.config):
        print(f"FATAL ERROR: Configuration file '{args.config}' not found.")
        return 1

    try:
        start_time = datetime.now()
        print(f"Starting Electrolyzer rolling horizon optimization at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        manager = ElectrolyzerRollingHorizonManager(args.config)
        results = manager.run_simulation()

        end_time = datetime.now()
        elapsed = end_time - start_time
        print(f"\nElectrolyzer rolling horizon simulation completed in {elapsed.total_seconds():.1f} seconds ({elapsed})")
        
        return 0

    except Exception as e:
        print(f"ERROR: An unexpected error occurred during the simulation: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
