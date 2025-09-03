# simple_market_clearing_eg.py exclusive group bids market clearing

"""
Market Clearing Module for Exclusive Group Bids (EGs).
This module is generic and can be used for any asset type.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class SimpleMarketClearingEG:
    """Implements market clearing logic for Exclusive Group Bids."""

    def __init__(self, config=None):
        self.config = config or {}
        self._set_defaults()
        # Get target variable from config, default to 'Price'
        self.target_variable = self.config.get('forecast', {}).get('target_variable', 'Price')

    def _set_defaults(self):
        defaults = {
            'output_dir': './results_eg/',
            'save_results': True
        }
        for key, value in defaults.items():
            self.config.setdefault(key, value)

    def clear_market(self, submitted_eg_bids, actual_mcp_df, current_period_start_dt, num_hours_in_period):
        """
        Clears the market for Exclusive Group Bids by selecting the profile with the highest economic surplus.
        """
        if not submitted_eg_bids:
            print("Market ClearingEG: No EG bids provided.")
            return pd.DataFrame(), self._empty_summary(num_hours_in_period), [0.0] * num_hours_in_period

        if actual_mcp_df.empty:
            print("Market ClearingEG: No actual market prices provided.")
            return pd.DataFrame(), self._empty_summary(num_hours_in_period), [0.0] * num_hours_in_period

        # Prepare MCP data
        mcp_df = actual_mcp_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(mcp_df['Date']):
            mcp_df['Date'] = pd.to_datetime(mcp_df['Date'])
        if mcp_df['Date'].dt.tz is not None:
            mcp_df['Date'] = mcp_df['Date'].dt.tz_localize(None)
        
        period_end_dt = current_period_start_dt + timedelta(hours=num_hours_in_period - 1)
        relevant_mcp_df = mcp_df[
            (mcp_df['Date'] >= current_period_start_dt) & (mcp_df['Date'] <= period_end_dt)
        ].set_index('Date')

        if len(relevant_mcp_df) != num_hours_in_period:
            print(f"Market ClearingEG WARNING: MCP data incomplete. Expected {num_hours_in_period} hours, found {len(relevant_mcp_df)}.")
            return pd.DataFrame(), self._empty_summary(num_hours_in_period), [0.0] * num_hours_in_period

        best_surplus = -float('inf')
        accepted_profile_info = None
        
        # Evaluate each profile
        for profile_bid in submitted_eg_bids:
            hourly_mw = profile_bid['hourly_mw_values']
            if len(hourly_mw) != num_hours_in_period:
                continue

            total_mw = sum(hourly_mw)
            cost_of_profile = (hourly_mw * relevant_mcp_df[self.target_variable]).sum()
            value_of_profile = total_mw * profile_bid['profile_bid_price']
            surplus = value_of_profile - cost_of_profile

            if surplus > best_surplus:
                best_surplus = surplus
                accepted_profile_info = profile_bid
        
        # Process the accepted profile
        if accepted_profile_info and best_surplus >= 0:
            print(f"Market ClearingEG: Profile ID {accepted_profile_info['profile_id']} accepted with surplus {best_surplus:.2f}.")
            return self._format_accepted_bid(accepted_profile_info, relevant_mcp_df.reset_index(), current_period_start_dt, num_hours_in_period)
        else:
            print("Market ClearingEG: No profile accepted (all negative surplus or no valid bids).")
            return pd.DataFrame(), self._empty_summary(num_hours_in_period), [0.0] * num_hours_in_period

    def _format_accepted_bid(self, accepted_profile, mcp_df, start_dt, num_hours):
        hourly_mw = accepted_profile['hourly_mw_values']
        records = []
        for i in range(num_hours):
            timestamp = start_dt + timedelta(hours=i)
            mcp = mcp_df.loc[mcp_df['Date'] == timestamp, self.target_variable].iloc[0]
            records.append({
                'Date': timestamp, 'Hour': i + 1,
                'Accepted MW': hourly_mw[i], 'MCP': mcp, 'Cost': hourly_mw[i] * mcp,
                'Profile_ID': accepted_profile['profile_id'],
                'Profile_Bid_Price': accepted_profile['profile_bid_price']
            })
        
        accepted_df = pd.DataFrame(records)
        total_mw = accepted_df['Accepted MW'].sum()
        total_cost = accepted_df['Cost'].sum()
        summary = {
            'total_accepted_mw': total_mw, 'total_cost': total_cost,
            'avg_price': total_cost / total_mw if total_mw > 1e-6 else 0,
            'accepted_profile_id': accepted_profile['profile_id'],
            'operating_hours': (accepted_df['Accepted MW'] > 1e-3).sum()
        }
        return accepted_df, summary, hourly_mw

    def _empty_summary(self, num_hours):
        return {'total_accepted_mw': 0, 'total_cost': 0, 'avg_price': 0, 'accepted_profile_id': None, 'operating_hours': 0}
