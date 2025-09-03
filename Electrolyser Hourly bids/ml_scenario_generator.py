
"""
ML-based Scenario Generation Module

This module generates realistic electricity price scenarios based on ML forecasts.
It creates scenarios that represent the uncertainty in price forecasts, which are
then used for stochastic optimization of bidding strategies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from sklearn.cluster import KMeans
from scipy.stats import levy_stable, norm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class ScenarioGenerator:
    """Generates electricity price scenarios based on ML forecasts."""
    
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
        # --- NEW: Scenario realism defaults ---
        scenario_realism_defaults = {
            'enable_jump_control': True,
            'max_hourly_jump_eur': 500.0,
            'jump_smoothing_factor': 0.5
        }

        defaults = {
            'num_scenarios': 2000,           # Initial number of scenarios
            'num_reduced': 30,               # Number of scenarios after reduction
            'distribution': 'levy_stable',   # Options: 'levy_stable', 'normal'
            'alpha': 1.31,                   # Alpha parameter for levy_stable
            'beta': 0.16,                    # Beta parameter for levy_stable
            'base_scale_factor': 0.05,       # Base volatility scale factor for day 1
            'uncertainty_growth_rate': 0.5,  # Growth rate of uncertainty over days
            'correlation': True,             # Apply hour-to-hour correlation
            'random_seed': 42,               # Random seed for reproducibility
            'output_dir': './scenarios/',    # Directory to save scenario files
            'scenario_realism': scenario_realism_defaults # Add new section
        }
        
        # Deep merge defaults with provided config
        def merge_dicts(base, new):
            for key, value in new.items():
                if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                    merge_dicts(base[key], value)
                else:
                    base[key] = value
            return base

        self.config = merge_dicts(defaults, self.config)
    
    def generate_hourly_correlation_matrix(self, hours=24, adjacent_corr=0.971, decay_factor=0.004745):
        """
        Generate realistic correlation matrix for electricity prices.
        
        Args:
            hours: Number of hours
            adjacent_corr: Correlation between adjacent hours
            decay_factor: Decay factor for correlation between distant hours
            
        Returns:
            Correlation matrix
        """
        # Initialize correlation matrix
        corr_matrix = np.eye(hours)
        
        # Fill correlations based on distance between hours
        for i in range(hours):
            for j in range(hours):
                if i != j:
                    # Minimum distance (accounting for circular nature of day)
                    distance = min(abs(i - j), hours - abs(i - j))
                    # Exponential decay with distance
                    corr_matrix[i, j] = adjacent_corr * np.exp(-decay_factor * distance)
        
        # Ensure matrix is positive definite
        min_eig = np.min(np.linalg.eigvals(corr_matrix))
        if min_eig <= 1e-10:
            corr_matrix += np.eye(hours) * (abs(min_eig) + 1e-6)
            
        return corr_matrix
    
    def generate_scenarios(self, forecast_df, horizon_days):
        """
        Generate scenarios around the forecast.
        
        Args:
            forecast_df: DataFrame with price forecasts
            horizon_days: Number of days in horizon
            
        Returns:
            Array of scenario prices
        """
        np.random.seed(self.config['random_seed'])
        
        # taking forecast prices
        forecast_prices = forecast_df['Forecast_Price'].values
        total_hours = len(forecast_prices)
        
        
        expected_hours = horizon_days * 24
        if total_hours != expected_hours:
            print(f"Warning: Forecast has {total_hours} hours, expected {expected_hours}")
            # Pad or truncate to expected hours
            if total_hours < expected_hours:
                # Pad with last value
                padding = np.full(expected_hours - total_hours, forecast_prices[-1])
                forecast_prices = np.concatenate([forecast_prices, padding])
            else:
                # Truncate
                forecast_prices = forecast_prices[:expected_hours]
            total_hours = expected_hours
        
        # Initialize scenarios array
        num_scenarios = self.config['num_scenarios']
        scenarios = np.zeros((num_scenarios, total_hours))
        
        # Generate correlation matrix 
        if self.config['correlation']:
            hourly_corr = self.generate_hourly_correlation_matrix(hours=24)
            
            # Extend to multi-day correlation with decay
            full_corr = np.zeros((total_hours, total_hours))
            day_decay = 0.126  # Correlation decay between days this is not avaible in config and needs to be updated here only
            
            for day_i in range(horizon_days):
                for day_j in range(horizon_days):
                    i_start = day_i * 24
                    i_end = (day_i + 1) * 24
                    j_start = day_j * 24
                    j_end = (day_j + 1) * 24
                    
                    if day_i == day_j:
                        # Same day - use hourly correlation
                        full_corr[i_start:i_end, j_start:j_end] = hourly_corr
                    else:
                        # Different days - apply decay
                        day_distance = abs(day_i - day_j)
                        day_factor = np.exp(-day_decay * day_distance)
                        full_corr[i_start:i_end, j_start:j_end] = hourly_corr * day_factor
            
            # Ensure positive definite
            min_eig = np.min(np.linalg.eigvals(full_corr))
            if min_eig <= 1e-10:
                full_corr += np.eye(total_hours) * (abs(min_eig) + 1e-6)
                
            # Cholesky decomposition for correlated sampling
            chol_matrix = np.linalg.cholesky(full_corr)
        else:
            chol_matrix = None
        
        # Calculate volatility scales that grow with horizon
        scales = np.zeros(total_hours)
        for day in range(horizon_days):
            # Calculate uncertainty growth factor
            growth_factor = 1 + self.config['uncertainty_growth_rate'] * (1 - np.exp(-day * 0.5))
            
            # Set scales for this day's hours
            day_start = day * 24
            day_end = (day + 1) * 24
            day_forecast = forecast_prices[day_start:day_end]
            
            # Base scales proportional to forecast price
            base_scales = np.abs(day_forecast) * self.config['base_scale_factor'] * growth_factor
            
            # Higher volatility during peak hours
            hour_indices = np.arange(24)
            peak_hours = ((hour_indices >= 7) & (hour_indices < 10)) | ((hour_indices >= 17) & (hour_indices < 21))
            base_scales[peak_hours] *= 1.2
            
            # Ensure minimum scale
            scales[day_start:day_end] = np.maximum(base_scales, 1e-6)
        
        # Generate random variables based on selected distribution
        if self.config['distribution'] == 'levy_stable':
            rand_vars = levy_stable.rvs(
                alpha=self.config['alpha'],
                beta=self.config['beta'],
                scale=1.0,
                loc=0.0,
                size=(num_scenarios, total_hours),
                random_state=self.config['random_seed']
            )
        else:  # Normal distribution
            rand_vars = norm.rvs(
                loc=0.0,
                scale=1.0,
                size=(num_scenarios, total_hours),
                random_state=self.config['random_seed']
            )
        
        # Apply correlation if enabled
        if self.config['correlation'] and chol_matrix is not None:
            rand_vars = np.dot(rand_vars, chol_matrix.T)
        
        # Generate scenarios by applying scaled random variables to forecast
        for i in range(num_scenarios):
            scenarios[i] = forecast_prices + rand_vars[i] * scales

        # to prevent unrealistic jumps in scenario prices
        realism_cfg = self.config.get('scenario_realism', {})
        if realism_cfg.get('enable_jump_control', False):
            max_jump = realism_cfg.get('max_hourly_jump_eur', 500.0)
            smoothing = realism_cfg.get('jump_smoothing_factor', 0.5)
            jumps_corrected = 0
            
            for i in range(num_scenarios):
                for t in range(1, total_hours):
                    jump = scenarios[i, t] - scenarios[i, t - 1]
                    if abs(jump) > max_jump:
                        jumps_corrected += 1
                        corrected_jump = np.sign(jump) * (max_jump + (abs(jump) - max_jump) * smoothing)
                        scenarios[i, t] = scenarios[i, t - 1] + corrected_jump
            
            if jumps_corrected > 0:
                print(f"  Jump Control: Corrected {jumps_corrected} unrealistic hourly price jumps (threshold: {max_jump} EUR).")

        # Ensure realistic prices (clip to reasonable bounds)
        scenarios = np.clip(scenarios, -500, 4000)
        
        print(f"Generated {num_scenarios} scenarios with {total_hours} hours each")
        return scenarios, forecast_prices
    
    def reduce_scenarios(self, scenarios):
        """
        Reduce number of scenarios using K-Means clustering.
        
        Args:
            scenarios: Array of scenario prices
            
        Returns:
            Tuple of (reduced_scenarios, probabilities)
        """
        np.random.seed(self.config['random_seed'])
        
        num_reduced = self.config['num_reduced']
        num_original = scenarios.shape[0]
        
        if num_reduced >= num_original:
            print(f"No reduction needed: requested {num_reduced} scenarios, have {num_original}")
            return scenarios, np.ones(num_original) / num_original
        
        # Apply K-Means clustering
        kmeans = KMeans(
            n_clusters=num_reduced,
            random_state=self.config['random_seed'],
            n_init='auto' # Use modern default
        )
        
        cluster_labels = kmeans.fit_predict(scenarios)
        
        # Get reduced scenarios (centroids)
        reduced_scenarios = kmeans.cluster_centers_
        
        # Calculate probabilities based on cluster sizes
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        probabilities = counts / num_original
        
        print(f"Reduced from {num_original} to {num_reduced} scenarios")
        return reduced_scenarios, probabilities
    
    def create_scenario_dataframe(self, reduced_scenarios, probabilities, start_date, horizon_days):
        """
        Create DataFrame with scenarios in the format expected by stochastic optimization.
        
        Args:
            reduced_scenarios: Array of reduced scenarios
            probabilities: Array of scenario probabilities
            start_date: Start date for the scenarios
            horizon_days: Number of days in horizon
            
        Returns:
            DataFrame with scenarios
        """
        n_scenarios = len(probabilities)
        total_hours = reduced_scenarios.shape[1]
        
        # Create hour column names (Hour_01, Hour_02, etc.)
        hour_cols = [f"Hour_{h+1:02d}" for h in range(total_hours)]
        
        # Create base DataFrame
        scenario_df = pd.DataFrame({
            "Scenario_ID": range(1, n_scenarios + 1),
            "Scenario_Probability": probabilities
        })
        
        # Add scenario prices
        for i, col_name in enumerate(hour_cols):
            scenario_df[col_name] = reduced_scenarios[:, i]
            
        return scenario_df
    
    def generate_and_save(self, forecast_df, start_date, horizon_days, output_file=None):
        """
        Generate scenarios, reduce them, and save to file.
        
        Args:
            forecast_df: DataFrame with price forecasts
            start_date: Start date string or datetime
            horizon_days: Number of days in horizon
            output_file: Optional output file path
            
        Returns:
            DataFrame with scenarios and path to saved file
        """
        # Convert start_date to datetime if it's a string
        if isinstance(start_date, str):
            start_dt = pd.to_datetime(start_date)
        else:
            start_dt = start_date
            
        # Generate scenarios
        scenarios, forecast_prices = self.generate_scenarios(forecast_df, horizon_days)
        
        # Reduce scenarios
        reduced_scenarios, probabilities = self.reduce_scenarios(scenarios)
        
        # Create DataFrame
        scenario_df = self.create_scenario_dataframe(reduced_scenarios, probabilities, start_dt, horizon_days)
        
        # Determine output file if not provided
        if output_file is None:
            if not os.path.exists(self.config['output_dir']):
                os.makedirs(self.config['output_dir'])
            
            start_str = start_dt.strftime('%Y%m%d')
            output_file = os.path.join(self.config['output_dir'], f"scenarios_{start_str}_{horizon_days}d.csv")
        
        # Save to file
        scenario_df.to_csv(output_file, index=False)
        print(f"Saved {len(scenario_df)} scenarios to {output_file}")
        
        return scenario_df, output_file
    
    def plot_scenarios(self, scenarios, forecast_prices=None, start_date=None, output_file=None):
        """
        Plot scenarios with forecast.
        NOT USED ANYMORE
        """
        plt.figure(figsize=(12, 6))
        
        # Generate x-axis values
        total_hours = scenarios.shape[1]
        if start_date is not None:
            if isinstance(start_date, str):
                start_dt = pd.to_datetime(start_date)
            else:
                start_dt = start_date
            date_range = pd.date_range(start=start_dt, periods=total_hours, freq='H')
            x_values = date_range
        else:
            x_values = np.arange(total_hours)
        
        # Plot scenarios with transparency
        for i in range(min(50, scenarios.shape[0])):  # Plot at most 50 for clarity
            plt.plot(x_values, scenarios[i], color='gray', alpha=0.1)
        
        # Calculate and plot percentiles
        percentiles = [10, 25, 50, 75, 90]
        colors = ['blue', 'green', 'red', 'green', 'blue']
        
        for p, color in zip(percentiles, colors):
            p_values = np.percentile(scenarios, p, axis=0)
            if p == 50:
                plt.plot(x_values, p_values, color=color, linewidth=2, label=f'P{p}')
            else:
                plt.plot(x_values, p_values, color=color, linewidth=1, label=f'P{p}')
                
        # Plot forecast if provided
        if forecast_prices is not None:
            plt.plot(x_values, forecast_prices, 'k--', linewidth=2, label='Forecast')
        
        plt.title('Price Scenarios with Percentiles')
        plt.xlabel('Time')
        plt.ylabel('Price (€/MWh)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis for datetime
        if isinstance(x_values[0], pd.Timestamp):
            plt.gcf().autofmt_xdate()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved scenario plot to {output_file}")
            
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == '__main__':
    from ml_forecaster import ElectricityPriceForecaster
    
    # Step 1: Generate a forecast
    forecaster = ElectricityPriceForecaster()
    data_file = 'scaled_data_NL4.xlsx'
    start_date = '2022-01-15'
    horizon_days = 5
    
    forecast_df = forecaster.forecast(data_file, start_date, horizon_days)
    
    # Step 2: Generate scenarios from the forecast
    scenario_config = {
        'num_scenarios': 2000,
        'num_reduced': 30,
        'distribution': 'levy_stable',
        'alpha': 1.31,
        'beta': 0.16,
        'base_scale_factor': 0.05,
        'uncertainty_growth_rate': 0.5,
        'correlation': True,
        'random_seed': 42,
        'output_dir': './scenarios/',
        'scenario_realism': {
            'enable_jump_control': True,
            'max_hourly_jump_eur': 500.0,
            'jump_smoothing_factor': 0.5
        }
    }
    
    generator = ScenarioGenerator(scenario_config)
    scenario_df, output_file = generator.generate_and_save(forecast_df, start_date, horizon_days)
    
    # Step 3: Plot some scenarios
    scenarios, forecast_prices = generator.generate_scenarios(forecast_df, horizon_days)
    generator.plot_scenarios(scenarios, forecast_prices, start_date, 'scenario_plot.png')
