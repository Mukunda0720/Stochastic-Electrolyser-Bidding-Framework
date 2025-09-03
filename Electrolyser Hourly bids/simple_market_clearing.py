# simple_market_clearing.py

"""
Simple Market Clearing Module (Updated)

Implements simplified market clearing logic for electricity markets.
Applies the rule: if bid price >= MCP, the quantity is accepted.
Removed H2-specific calculations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

class SimpleMarketClearing:
    """Implements simplified market clearing logic for electricity markets."""

    def __init__(self, config=None):
        """
        Initialize with configuration parameters.

        Args:
            config: Dictionary with configuration parameters
        """
        self.config = config or {}
        self._set_defaults()

    def _set_defaults(self):
        """Set default configuration parameters."""
        defaults = {
            'output_dir': './results/', # Default output directory
            'save_results': True       # Whether to save results files
        }
        # Merge defaults with provided config
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value

    def clear_market(self, bids_df, actual_mcp_df):
        """
        Clear the market based on simple price taker assumption.

        Args:
            bids_df: DataFrame with bids (must have 'Date', 'Bid Price', 'Bid Quantity (MW)')
            actual_mcp_df: DataFrame with actual prices (must have 'Date', 'Price')

        Returns:
            DataFrame with accepted bids
        """
        if bids_df.empty:
            print("Market Clearing: No bids provided.")
            return pd.DataFrame()

        if actual_mcp_df.empty:
            print("Market Clearing: No actual market prices provided.")
            return pd.DataFrame()

        # Ensure date columns are datetime
        try:
            if not pd.api.types.is_datetime64_any_dtype(bids_df['Date']):
                bids_df['Date'] = pd.to_datetime(bids_df['Date'])
            if not pd.api.types.is_datetime64_any_dtype(actual_mcp_df['Date']):
                actual_mcp_df['Date'] = pd.to_datetime(actual_mcp_df['Date'])
        except Exception as e:
            print(f"Error converting Date columns to datetime: {e}")
            return pd.DataFrame()

        accepted_bids = []

        # Merge bids with actual prices for efficient lookup
        try:
            # Use the correct column name for actual price (assuming 'Price')
            if 'Price' not in actual_mcp_df.columns:
                 raise ValueError("Actual MCP DataFrame must contain a 'Price' column.")

            merged_df = pd.merge(bids_df, actual_mcp_df[['Date', 'Price']], on='Date', how='left')
            merged_df.rename(columns={'Price': 'MCP'}, inplace=True) # Rename for clarity
        except Exception as e:
            print(f"Error merging bids and actual prices: {e}")
            return pd.DataFrame()

        # Check for missing MCPs after merge
        if merged_df['MCP'].isnull().any():
            missing_dates = merged_df.loc[merged_df['MCP'].isnull(), 'Date'].dt.strftime('%Y-%m-%d %H:%M').unique()
            print(f"Warning: Missing MCP for dates: {', '.join(missing_dates)}")
            # Filter out rows where MCP is missing
            merged_df = merged_df.dropna(subset=['MCP'])
            if merged_df.empty:
                 print("No bids remaining after removing those with missing MCPs.")
                 return pd.DataFrame()


        # Apply clearing rule: Bid Price >= MCP
        required_bid_cols = ['Bid Price', 'Bid Quantity (MW)', 'Date', 'MCP']
        if not all(col in merged_df.columns for col in required_bid_cols):
             missing_cols = [col for col in required_bid_cols if col not in merged_df.columns]
             print(f"ERROR: Missing required columns in bids DataFrame after merge: {missing_cols}")
             return pd.DataFrame()

        accepted_mask = merged_df['Bid Price'] >= merged_df['MCP']
        accepted_df = merged_df.loc[accepted_mask].copy()

        if not accepted_df.empty:
             # Calculate cost based on MCP
             accepted_df['Cost'] = accepted_df['Bid Quantity (MW)'] * accepted_df['MCP']
             # Rename columns for consistency
             accepted_df.rename(columns={'Bid Quantity (MW)': 'Accepted MW'}, inplace=True)
             # Add Hour and Day if not present from bids_df
             if 'Hour' not in accepted_df.columns:
                  accepted_df['Hour'] = accepted_df['Date'].dt.hour + 1
             if 'Day' not in accepted_df.columns and not accepted_df.empty:
                  # Calculate Day relative to the minimum date in the accepted bids
                  min_date = accepted_df['Date'].min().date()
                  accepted_df['Day'] = accepted_df['Date'].apply(lambda d: (d.date() - min_date).days + 1)

             # Select and order columns for output
             output_cols = ['Date', 'Hour', 'Day', 'Bid Price', 'MCP', 'Accepted MW', 'Cost']
             # Add optional columns if they exist
             if 'Block' in accepted_df.columns: output_cols.insert(3, 'Block')
             if 'Total Blocks' in accepted_df.columns: output_cols.insert(4, 'Total Blocks')

             # Filter to keep only existing columns in the final output
             final_output_cols = [col for col in output_cols if col in accepted_df.columns]
             accepted_df = accepted_df[final_output_cols]

        print(f"Market Clearing: Accepted {len(accepted_df)} out of {len(bids_df)} bid records.")
        return accepted_df


    def calculate_summary(self, accepted_df):
        """
        Calculate summary statistics for accepted bids.

        Args:
            accepted_df: DataFrame with accepted bids (output from clear_market)

        Returns:
            Dictionary with summary statistics (total power, cost, avg price, operating hours)
        """
        if accepted_df is None or accepted_df.empty:
            return {
                'total_accepted_mw': 0.0,
                'total_cost': 0.0,
                'avg_price': 0.0,
                'operating_hours': 0
            }

        total_accepted_mw = accepted_df['Accepted MW'].sum()
        total_cost = accepted_df['Cost'].sum()
        # Calculate average price safely
        avg_price = total_cost / total_accepted_mw if total_accepted_mw > 1e-9 else 0.0
        # Count unique hours where power was accepted
        operating_hours = accepted_df['Date'].nunique() if not accepted_df.empty else 0

        summary = {
            'total_accepted_mw': total_accepted_mw,
            'total_cost': total_cost,
            'avg_price': avg_price,
            'operating_hours': operating_hours
        }

        return summary

    def save_results(self, accepted_df, summary, strategy_name, start_date_str):
        """
        Save results to files.

        Args:
            accepted_df: DataFrame with accepted bids
            summary: Dictionary with summary statistics
            strategy_name: Name of the bidding strategy or simulation
            start_date_str: Start date string (e.g., '20241201') for filenames
        """
        if not self.config.get('save_results', False):
            return

        output_dir = self.config.get('output_dir', './results/')
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Save accepted bids
            if accepted_df is not None and not accepted_df.empty:
                accepted_file = os.path.join(output_dir, f"accepted_bids_{strategy_name}_{start_date_str}.xlsx")
                accepted_df.to_excel(accepted_file, index=False)
                print(f"Saved accepted bids to {accepted_file}")

            # Save summary
            summary_df = pd.DataFrame([summary]) # Convert dict to DataFrame
            summary_file = os.path.join(output_dir, f"summary_{strategy_name}_{start_date_str}.xlsx")
            summary_df.to_excel(summary_file, index=False)
            print(f"Saved summary to {summary_file}")

        except Exception as e:
            print(f"Error saving market clearing results for {strategy_name}: {e}")


# Example usage (kept for testing)
if __name__ == '__main__':
    # Create sample bid data (single block per hour for simplicity)
    bid_dates = pd.date_range(start='2024-01-15', periods=24, freq='H')
    bids_data = []
    for i, date in enumerate(bid_dates):
        bids_data.append({
            'Date': date, 'Bid Price': 100.0, 'Bid Quantity (MW)': 50.0 # Example bid
        })
    bids_df = pd.DataFrame(bids_data)

    # Create sample MCP data
    mcp_data = [{'Date': date, 'Price': 80.0 + i*2} for i, date in enumerate(bid_dates)] # Example prices
    mcp_df = pd.DataFrame(mcp_data)

    # Initialize and run market clearing
    market = SimpleMarketClearing({'output_dir': './market_clearing_test'})
    accepted_df = market.clear_market(bids_df, mcp_df)
    summary = market.calculate_summary(accepted_df)

    print("\nAccepted Bids:")
    print(accepted_df.head())
    print("\nSummary:")
    print(summary)

    # Save results
    market.save_results(accepted_df, summary, 'test_strategy', '20240115')
