# ml_forecaster.py

"""
Improved Prophet-based Price Forecaster Module for Electricity Markets

This module uses Facebook/Meta's Prophet time series forecasting model to predict
electricity prices with high accuracy while avoiding overfitting. It implements
robust cross-validation, adaptive regularization, and careful feature selection
to ensure good generalization performance.

"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import Prophet for time series forecasting
try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    from prophet.serialize import model_to_json, model_from_json
    PROPHET_AVAILABLE = True
except ImportError:
    print("Prophet library not found. Install with: pip install prophet")
    PROPHET_AVAILABLE = False


class OutlierHandler:
    """
    Robust outlier detection and handling for price data.
    Uses median absolute deviation for better robustness than standard deviation. but follows the three siigma rule
    """

    def __init__(self, method='clip', threshold=3):
        """
        Initialize the outlier handler.
        
        Args:
            method (str): Method for handling outliers ('clip' or 'winsorize')
            threshold (float): Number of MADs for outlier detection
        """
        self.method = method
        self.threshold = threshold
        self.lower_bound = None
        self.upper_bound = None

    def fit(self, series):
        """
        Calculate outlier bounds based on median absolute deviation.
        
        Args:
            series: Series of values to analyze
            
        Returns:
            self: For method chaining
        """
        # Calculate bounds using median absolute deviation (robust to outliers)
        median = series.median()
        
        # Calculate MAD, handle cases where median is 0 or series is constant
        mad = np.median(np.abs(series - median))
        if mad < 1e-6:  # If MAD is very small (e.g., constant series)
            mad = series.std()  # Use standard deviation as fallback
            if mad < 1e-6:  # If std is also zero, no spread
                # Set bounds very close to the median to avoid issues
                self.lower_bound = median - 1e-3
                self.upper_bound = median + 1e-3
                return self

        # Use Median Absolute Deviation for robust outlier detection
        # 1.4826 is the factor to make MAD comparable to standard deviation for normal distribution
        self.lower_bound = median - self.threshold * 1.4826 * mad
        self.upper_bound = median + self.threshold * 1.4826 * mad
        return self

    def transform(self, series):
        """
        Apply outlier handling to a series.
        
        Args:
            series: Series of values to transform
            
        Returns:
            Transformed series with outliers handled
        """
        if self.lower_bound is None or self.upper_bound is None:
            print("Warning: Outlier bounds not fitted. Returning original series.")
            return series

        if self.method == 'clip':
            return series.clip(self.lower_bound, self.upper_bound)
        elif self.method == 'winsorize':
            
            return np.where(series < self.lower_bound, self.lower_bound,
                            np.where(series > self.upper_bound, self.upper_bound, series))
        return series


class ElectricityPriceForecaster:
    """
    
    Features cross-validation, adaptive regularization, and focused feature selection
    to avoid overfitting and improve generalization performance.
    """

    def __init__(self, config=None):
        """
        Initialize the forecaster with configuration settings.

        Args:
            config: Dictionary with configuration parameters
        """
        self.config = config or {}
        self._set_defaults()
        self.model = None
        self.feature_engineering_params = {}  # Store means/stds for normalization
        self.regressors_list = []  # Store names of regressors added
        self.market_floor = self.config.get('prophet', {}).get('floor', -500.0)
        self.market_cap = self.config.get('prophet', {}).get('cap', 4000.0)
        self.target_variable = self.config.get('target_variable', 'Price')
        self.cv_metrics = None  # Store cross-validation metrics
        self.feature_importances = {}  # Store feature importance scores
        
    def _set_defaults(self):
        """Set default configuration parameters with conservative choices to avoid overfitting."""
        # Default Prophet settings (conservative to prevent overfitting)
        prophet_defaults = {
            'changepoint_prior_scale': 0.05,    # Default, conservative setting
            'seasonality_prior_scale': 10.0,    # Moderate regularization for seasonality
            'holidays_prior_scale': 10.0,       # Moderate regularization for holidays
            'daily_seasonality': True,          # Daily patterns are important for electricity
            'weekly_seasonality': True,         # Weekly patterns are important for electricity
            'yearly_seasonality': 'auto',       # Let Prophet decide based on data length
            'seasonality_mode': 'multiplicative', # Often better for electricity prices
            'interval_width': 0.95,             # 95% prediction interval
            'mcmc_samples': 0,                  # Use MAP estimation (faster)
            'country_code': 'NL',               # Country code for holidays
            'floor': -500.0,                    # Market floor
            'cap': 4000.0,                      # Market cap
        }

        # Cross-validation settings
        cv_defaults = {
            'enabled': True,                   # Enable cross-validation
            'initial': '30 days',              # Initial training period
            'period': '7 days',                # Spacing between cutoff dates
            'horizon': '7 days',               # Forecast horizon for CV
            'parallel': 'processes',           # Use parallel processing
        }

        # Feature selection defaults
        feature_selection_defaults = {
            'enabled': True,                   # Enable feature selection
            'max_features': 10,                # Maximum number of features to use
            'correlation_threshold': 0.85,     # Threshold for removing highly correlated features
        }

        defaults = {
            'target_variable': 'Price',
            'training_days': 90,               # Default training window (90 days is conservative)
            'model_path': './models/',
            'retrain': True,
            'prophet': prophet_defaults,
            'cv': cv_defaults,                 # Add cross-validation settings
            'feature_selection': feature_selection_defaults,  # Add feature selection settings
            'feature_sets': [
                'hour',                        # Keep time components
                'day_of_week',                 # Weekly patterns
                'month',                       # Monthly patterns
                'load',                        # Load is typically important
                'solar_wind'                   # Renewable generation
            ],
            'outlier_handling': {
                'enabled': True,
                'method': 'clip',
                'threshold': 3
            },
            'market_input_file': 'scaled_data_NL4.xlsx'  # Default data file
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
        
        # Update market floor/cap from final merged config
        self.market_floor = self.config.get('prophet', {}).get('floor', -500.0)
        self.market_cap = self.config.get('prophet', {}).get('cap', 4000.0)
        
        # Update target variable name
        self.target_variable = self.config.get('target_variable', 'Price')

    def load_data(self, data_file):
        """
        Load historical data from file with robust error handling.

        Args:
            data_file: Path to the data file (Excel or CSV)

        Returns:
            DataFrame with loaded data, or empty DataFrame on error
        """
        try:
            if data_file.endswith('.xlsx'):
                data = pd.read_excel(data_file)
            elif data_file.endswith('.csv'):
                data = pd.read_csv(data_file)
            else:
                raise ValueError(f"Unsupported file format: {data_file}. Please use .xlsx or .csv.")

            if 'Date' not in data.columns:
                raise ValueError("Data file must contain a 'Date' column.")
            if self.target_variable not in data.columns:
                raise ValueError(f"Data file must contain the target variable column '{self.target_variable}'.")

            # Convert to datetime 
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.sort_values(by='Date').reset_index(drop=True)

            # Handle missing values (forward fill then backward fill)
            initial_missing = data.isnull().sum().sum()
            if initial_missing > 0:
                data = data.fillna(method='ffill').fillna(method='bfill')
                print(f"Filled {initial_missing} missing values using ffill/bfill.")
                if data.isnull().sum().sum() > 0:
                    print("Warning: Some missing values remain after ffill/bfill.")

            # Clip prices to market bounds 
            original_min = data[self.target_variable].min()
            original_max = data[self.target_variable].max()
            data[self.target_variable] = data[self.target_variable].clip(self.market_floor, self.market_cap)
            if data[self.target_variable].min() > original_min or data[self.target_variable].max() < original_max:
                print(f"Clipped '{self.target_variable}' data to market bounds [{self.market_floor}, {self.market_cap}].")

            print(f"Loaded {len(data)} rows from {data_file}. "
                  f"Date range: {data['Date'].min()} to {data['Date'].max()}")
            return data

        except FileNotFoundError:
            print(f"Error: Data file not found at {data_file}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading data from {data_file}: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def _apply_feature_engineering(self, df):
        """
        Apply feature engineering with additional predictive features
        specifically designed for electricity price forecasting.
        
        Args:
            df: Input DataFrame with 'Date' column
            
        Returns:
            Tuple of (DataFrame with features, holiday DataFrame or None)
        """
        df_eng = df.copy()
        
        # ===== Basic time features =====
        df_eng['Hour'] = df_eng['Date'].dt.hour
        df_eng['DayOfWeek'] = df_eng['Date'].dt.dayofweek
        df_eng['Month'] = df_eng['Date'].dt.month
        df_eng['WeekOfYear'] = df_eng['Date'].dt.isocalendar().week.astype(int)
        df_eng['DayOfYear'] = df_eng['Date'].dt.dayofyear
        df_eng['IsWeekend'] = df_eng['DayOfWeek'].isin([5, 6]).astype(int)
        
        #Add day of month feature  mostly not needed or used 
        df_eng['DayOfMonth'] = df_eng['Date'].dt.day
        
        df_eng['Quarter'] = df_eng['Date'].dt.quarter
        
        #feature for time of day will play a cruicial role in electricity prices
        df_eng['IsMorningPeak'] = ((df_eng['Hour'] >= 7) & (df_eng['Hour'] <= 9)).astype(int)
        df_eng['IsEveningPeak'] = ((df_eng['Hour'] >= 17) & (df_eng['Hour'] <= 20)).astype(int)

        # National holidays can show change in price patterns
        holiday_events = None
        try:
            import holidays
            country_code = self.config.get('prophet', {}).get('country_code', 'NL')
            years = df_eng['Date'].dt.year.unique()
            
            # Get holidays for the relevant country
            try:
                country_holidays_dict = holidays.country_holidays(country_code, years=years)
            except TypeError:  # For older versions of the holidays package
                country_holidays_dict = holidays.CountryHoliday(country_code, years=years)

            # Mark holiday dates
            df_eng['IsHoliday'] = df_eng['Date'].dt.date.apply(
                lambda date: date in country_holidays_dict).astype(int)
                
            # New: Add working day indicator (not weekend and not holiday)
            df_eng['IsWorkingDay'] = ((df_eng['IsWeekend'] == 0) & 
                                     (df_eng['IsHoliday'] == 0)).astype(int)

            # Create holiday DataFrame for Prophet
            holiday_list = []
            for date, name in sorted(country_holidays_dict.items()):
                # Ensure holidays are within the dataframe's date range
                if date >= df_eng['Date'].min().date() and date <= df_eng['Date'].max().date():
                    holiday_list.append({
                        'ds': pd.to_datetime(date), 
                        'holiday': name,
                        'lower_window': 0,  # Same day effect
                        'upper_window': 1    # Holiday effect might linger to next day
                    })

            if holiday_list:
                holiday_events = pd.DataFrame(holiday_list)
                print(f"Generated {len(holiday_events)} holiday events for Prophet.")

        except ImportError:
            print("Optional library 'holidays' not found. Skipping holiday features.")
            df_eng['IsHoliday'] = 0
            df_eng['IsWorkingDay'] = (df_eng['IsWeekend'] == 0).astype(int)

        # Price lags 
        target_var = self.target_variable
        lag_features = []
        
        if 'price_lags' in self.config.get('feature_sets', []):
            # Essential lags for electricity markets
            for lag in [24, 48, 168]:  # 1 day, 2 days, 1 week
                lag_col = f'{target_var}_lag{lag}'
                df_eng[lag_col] = df_eng[target_var].shift(lag)
                lag_features.append(lag_col)
                
            # hour-to-hour lag (captures very recent price momentum)
            df_eng[f'{target_var}_lag1'] = df_eng[target_var].shift(1)
            lag_features.append(f'{target_var}_lag1')
            
            # price momentum indicators
            # Day-ahead momentum (today vs. yesterday)
            df_eng[f'{target_var}_diff24'] = df_eng[target_var] - df_eng[f'{target_var}_lag24']
            lag_features.append(f'{target_var}_diff24')
            
            # Week-on-week change (this weekday vs. same weekday last week)
            df_eng[f'{target_var}_diff168'] = df_eng[target_var] - df_eng[f'{target_var}_lag168']
            lag_features.append(f'{target_var}_diff168')
            
            # Hour-to-hour change (price acceleration/deceleration)
            df_eng[f'{target_var}_diff1'] = df_eng[target_var].diff()
            lag_features.append(f'{target_var}_diff1')
            
            # New: Weekend interaction feature (captures different price patterns on weekends)
            weekend_interaction = f'{target_var}_weekend_effect'
            df_eng[weekend_interaction] = df_eng[f'{target_var}_lag24'] * df_eng['IsWeekend']
            lag_features.append(weekend_interaction)
            
            # Hour-of-day interaction (captures hour-specific price patterns)
            for peak_hour in [8, 12, 18]:  # Key hours (morning peak, mid-day, evening peak)
                hour_mask = (df_eng['Hour'] == peak_hour).astype(int)
                hour_interaction = f'{target_var}_hour{peak_hour}_effect'
                df_eng[hour_interaction] = df_eng[f'{target_var}_lag24'] * hour_mask
                lag_features.append(hour_interaction)

        if 'price_stats' in self.config.get('feature_sets', []):
            # Rolling statistics for different windows
            for window in [24, 168]:  # Daily and weekly windows
                # Rolling mean (captures local average price level)
                mean_col = f'{target_var}_roll_mean{window}'
                df_eng[mean_col] = df_eng[target_var].shift(1).rolling(window=window).mean()
                lag_features.append(mean_col)
                
                # Rolling standard deviation (captures local price volatility)
                std_col = f'{target_var}_roll_std{window}'
                df_eng[std_col] = df_eng[target_var].shift(1).rolling(window=window).std()
                lag_features.append(std_col)
                
                # New: Rolling min and max (captures price extremes)
                min_col = f'{target_var}_roll_min{window}'
                df_eng[min_col] = df_eng[target_var].shift(1).rolling(window=window).min()
                lag_features.append(min_col)
                
                max_col = f'{target_var}_roll_max{window}'
                df_eng[max_col] = df_eng[target_var].shift(1).rolling(window=window).max()
                lag_features.append(max_col)
                
                # New: Rolling median (robust to outliers)
                median_col = f'{target_var}_roll_median{window}'
                df_eng[median_col] = df_eng[target_var].shift(1).rolling(window=window).median()
                lag_features.append(median_col)

            # Calculate price volatility ratio (helps identify volatile periods)
            df_eng['price_volatility_ratio'] = df_eng[f'{target_var}_roll_std24'] / (df_eng[f'{target_var}_roll_mean24'] + 1e-8)
            lag_features.append('price_volatility_ratio')

        # Fill NaNs created by lags/rolling features with sensible values
        if lag_features:
            # Forward fill first (use previous values)
            df_eng[lag_features] = df_eng[lag_features].fillna(method='ffill')
            # Then backward fill any remaining NaNs (use future values)
            df_eng[lag_features] = df_eng[lag_features].fillna(method='bfill')
            # For any remaining NaNs, fill with column-specific means
            for feature in lag_features:
                if feature in df_eng.columns and df_eng[feature].isnull().any():
                    df_eng[feature] = df_eng[feature].fillna(df_eng[feature].mean())

        return df_eng, holiday_events

    def _select_important_features(self, df, target_col):
        """
        Select the most important features using correlation
        
        Args:
            df: DataFrame with features
            target_col: Target column name
            
        Returns:
            List of selected feature names
        """
        feature_selection_config = self.config.get('feature_selection', {})
        if not feature_selection_config.get('enabled', True):
            # If feature selection is disabled, return all potential regressor columns
            return [col for col in df.columns if col not in ['Date', 'ds', 'y', target_col, 'floor', 'cap']]
        
        # Extract config parameters
        max_features = feature_selection_config.get('max_features', 10)
        correlation_threshold = feature_selection_config.get('correlation_threshold', 0.85)
        
        # Calculate correlation with target
        potential_features = [col for col in df.columns 
                              if col not in ['Date', 'ds', 'y', target_col, 'floor', 'cap']]
        
        if not potential_features:
            return []
            
        # Calculate correlations
        corr_with_target = {}
        for feature in potential_features:
            # Skip non-numeric features
            if not pd.api.types.is_numeric_dtype(df[feature]):
                continue
                
            # Calculate correlation, handling potential errors
            try:
                correlation = df[feature].corr(df[target_col])
                if pd.notnull(correlation):  # Only store if not NaN
                    corr_with_target[feature] = abs(correlation)  # Use absolute correlation
            except:
                continue
        
        # Sort features by correlation (descending)
        sorted_features = sorted(corr_with_target.items(), key=lambda x: x[1], reverse=True)
        
        # Critical features for electricity prices that should be included if available
        critical_features = [
            'Load_Forecast', 'Solar_Wind_Forecast',  # Supply-demand fundamentals
            f'{target_col}_lag24', f'{target_col}_lag168',  # Key lags (day-ahead, week-ahead)
            'Hour', 'DayOfWeek', 'IsWeekend'  # Essential time features
        ]
        
        
        critical_features = [f for f in critical_features if f in df.columns and f in corr_with_target]
        
        # Add critical features first, then add other selected features up to max_features
        final_features = []
        for feature in critical_features:
            if feature not in final_features:
                final_features.append(feature)
                # Calculate and store importance 
                if feature not in self.feature_importances and feature in corr_with_target:
                    self.feature_importances[feature] = corr_with_target[feature]
        
        # Select additional features
        for feature, corr in sorted_features:
            # Skip if we already have enough features
            if len(final_features) >= max_features:
                break
                
            # Skip if already added as a critical feature
            if feature in final_features:
                continue
                
            # Check correlation with already selected features
            should_add = True
            for selected in final_features:
                try:
                    # Calculate correlation between this feature and already selected feature
                    feature_corr = abs(df[feature].corr(df[selected]))
                    if feature_corr > correlation_threshold:
                        should_add = False
                        break
                except:
                    continue
                    
            if should_add:
                final_features.append(feature)
                self.feature_importances[feature] = corr  # Store importance
        
        print(f"Selected {len(final_features)} features from {len(potential_features)} candidates")
        if final_features:
            print("Top features (absolute correlation with target):")
            for feature in final_features[:5]:  # Show top 5
                print(f"  - {feature}: {self.feature_importances.get(feature, 'N/A'):.4f}")
        
        return final_features

    def preprocess_data(self, data, fit_transform_outliers=True):
        """
        Preprocess data: feature engineering and outlier handling.

        Args:
            data: DataFrame with historical data
            fit_transform_outliers: Whether to fit the outlier handler and transform
                                    Set to False when transforming future data

        Returns:
            Tuple: (Preprocessed DataFrame, holiday events DataFrame or None)
        """
        df = data.copy()

        # Apply feature engineering (time features, holidays, lags, etc.)
        df_eng, holiday_events = self._apply_feature_engineering(df)

        # Handle outliers in the target variable if enabled
        target_var = self.target_variable
        outlier_config = self.config.get('outlier_handling', {})
        if outlier_config.get('enabled', False):
            outlier_method = outlier_config.get('method', 'clip')
            outlier_threshold = outlier_config.get('threshold', 3)

            if fit_transform_outliers:
                print(f"Fitting and applying outlier handling ('{outlier_method}', threshold={outlier_threshold}).")
                self.outlier_handler = OutlierHandler(method=outlier_method, threshold=outlier_threshold)
                df_eng[target_var] = self.outlier_handler.fit(df_eng[target_var]).transform(df_eng[target_var])
            elif hasattr(self, 'outlier_handler') and self.outlier_handler is not None:
                print("Applying existing outlier handler.")
                df_eng[target_var] = self.outlier_handler.transform(df_eng[target_var])
            else:
                print("Warning: Outlier handling enabled but no handler fitted/loaded. Skipping.")

        return df_eng, holiday_events

    def _prepare_prophet_df(self, data, for_training=True):
        """
        Prepare data for Prophet 
        
        Args:
            data: Preprocessed DataFrame
            for_training: Whether this is for training (enables feature selection)
            
        Returns:
            DataFrame formatted for Prophet
        """
        df_prophet = data[['Date']].rename(columns={'Date': 'ds'})
        target_var = self.target_variable
        df_prophet['y'] = data[target_var]

        # Add floor and cap for growth trend in prophet model 
        df_prophet['floor'] = self.market_floor
        df_prophet['cap'] = self.market_cap

        # Select important features if this is for training
        selected_features = []
        if for_training:
            selected_features = self._select_important_features(data, target_var)
            self.regressors_list = selected_features.copy()
        else:
            # Use previously selected features
            selected_features = self.regressors_list
            
        # Reset feature engineering params
        self.feature_engineering_params = {}
        
        # Add selected features as regressors
        for feature in selected_features:
            if feature in data.columns:
                # For certain features that should be normalized
                if feature in ['Load_Forecast', 'Solar_Wind_Forecast', 'Total_Generation_Forecast']:
                    # Normalize the feature
                    mean = data[feature].mean()
                    std = data[feature].std()
                    
                    # Store normalization parameters
                    self.feature_engineering_params[f'{feature}_mean'] = mean
                    self.feature_engineering_params[f'{feature}_std'] = std
                    
                    norm_col_name = f'{feature}_normalized'
                    if std > 1e-6:  # Avoid division by zero
                        df_prophet[norm_col_name] = (data[feature] - mean) / std
                    else:
                        df_prophet[norm_col_name] = 0.0
                        
                    # Update the feature name in the list
                    idx = self.regressors_list.index(feature) if feature in self.regressors_list else -1
                    if idx >= 0:
                        self.regressors_list[idx] = norm_col_name
                else:
                    # Add feature directly
                    df_prophet[feature] = data[feature]
                    
        # Fill any missing values
        for col in df_prophet.columns:
            if col not in ['ds', 'y', 'floor', 'cap'] and df_prophet[col].isnull().any():
                # Fill with mean for numeric columns
                if pd.api.types.is_numeric_dtype(df_prophet[col]):
                    fill_value = df_prophet[col].mean()
                    df_prophet[col].fillna(fill_value, inplace=True)
                    
        print(f"Prepared Prophet DataFrame with {len(df_prophet.columns)} columns")
        print(f"Using {len(self.regressors_list)} regressors: {', '.join(self.regressors_list[:5])}"
              f"{' and more...' if len(self.regressors_list) > 5 else ''}")
        
        return df_prophet

    def _configure_prophet_model(self):
        """
        Configure Prophet model with adaptive regularization based on data characteristics.
        
        Returns:
            Configured Prophet model
        """
        prophet_config = self.config.get('prophet', {})
        
        # Create base model
        model = Prophet(
            changepoint_prior_scale=prophet_config.get('changepoint_prior_scale', 0.05),
            seasonality_prior_scale=prophet_config.get('seasonality_prior_scale', 10.0),
            holidays_prior_scale=prophet_config.get('holidays_prior_scale', 10.0),
            daily_seasonality=False,  # We'll add it manually with custom settings
            weekly_seasonality=False, # We'll add it manually with custom settings
            yearly_seasonality=prophet_config.get('yearly_seasonality', 'auto'),
            seasonality_mode=prophet_config.get('seasonality_mode', 'multiplicative'),
            changepoint_range=prophet_config.get('changepoint_range', 0.9),
            interval_width=prophet_config.get('interval_width', 0.95),
            mcmc_samples=prophet_config.get('mcmc_samples', 0)
        )
        
        # Add daily seasonality with specific prior
        model.add_seasonality(
            name='daily',
            period=1,
            fourier_order=6,  # higher to capture complex daily patterns
            prior_scale=prophet_config.get('daily_seasonality', 5.0)
        )
        
        # Add weekly seasonality with specific prior
        model.add_seasonality(
            name='weekly',
            period=7,
            fourier_order=4,  # Lower to avoid overfitting
            prior_scale=prophet_config.get('weekly_seasonality', 10.0)
        )
        
        # Add regressors with improved prior scales
        for regressor in self.regressors_list:
            if regressor.endswith('_normalized'):
                model.add_regressor(regressor, prior_scale=3.0)  # Reduced from 5.0
            elif regressor.startswith('Is') or regressor in ['Hour', 'DayOfWeek', 'Month']:
                model.add_regressor(regressor, prior_scale=0.5)  # Increased from 0.1
            else:
                model.add_regressor(regressor, prior_scale=10.0)
        
        return model
        
    def _perform_cross_validation(self, model, df):
        """
        Perform cross-validation to evaluate model performance and detect overfitting during development.
        additional feature were added after running this
        Args:
            model: Fitted Prophet model
            df: Prophet DataFrame used for training
            
        Returns:
            DataFrame with cross-validation metrics
        """
        cv_config = self.config.get('cv', {})
        if not cv_config.get('enabled', True) or len(df) < 100:
            print("Cross-validation skipped (disabled or insufficient data)")
            return None
            
        try:
            # Extract CV parameters
            initial = cv_config.get('initial', '30 days')
            period = cv_config.get('period', '7 days')
            horizon = cv_config.get('horizon', '7 days')
            parallel = cv_config.get('parallel', 'processes')
            
            print(f"Performing cross-validation with {initial} initial, "
                  f"{period} period, {horizon} horizon")
                  
            # Run cross-validation
            cv_results = cross_validation(
                model=model,
                initial=initial,
                period=period,
                horizon=horizon,
                parallel=parallel
            )
            
            # Calculate performance metrics
            cv_metrics = performance_metrics(cv_results)
            
            # Store CV metrics 
            self.cv_metrics = cv_metrics
            
            # Print summary metrics
            print("\nCross-Validation Metrics:")
            for metric in ['mae', 'rmse', 'mape', 'coverage']:
                if metric in cv_metrics:
                    value = cv_metrics[metric].mean()
                    print(f"  {metric.upper()}: {value:.4f}")
                    
            return cv_metrics
            
        except Exception as e:
            print(f"Error during cross-validation: {e}")
            import traceback
            traceback.print_exc()
            return None

    def train_model(self, data_file, start_date=None):
        """
        Train Prophet forecasting model .

        Args:
            data_file: Path to historical data file
            start_date: Optional start date for the first forecast period

        Returns:
            Trained Prophet model or None on failure
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is required for this forecaster. Install with: pip install prophet")

        # Load and preprocess data
        full_data = self.load_data(data_file)
        if full_data.empty:
            raise ValueError("No data available for training.")

        processed_data, holiday_events = self.preprocess_data(full_data, fit_transform_outliers=True)

        # Select training window
        training_data = None
        if start_date is not None:
            start_dt = pd.to_datetime(start_date)
            training_end_dt = start_dt - timedelta(hours=1)
            training_days = self.config.get('training_days', 90)
            training_start_dt = training_end_dt - timedelta(days=training_days)

            # Filter processed data
            mask = (processed_data['Date'] >= training_start_dt) & (processed_data['Date'] <= training_end_dt)
            training_data = processed_data.loc[mask].copy()
            print(f"Training on data from {training_data['Date'].min()} to {training_data['Date'].max()} "
                  f"({len(training_data)} rows)")
        else:
            # If no start date, train on all available processed data please mention start date in config to prevent long training times
            training_data = processed_data.copy()
            print(f"No start_date provided, training on all available data ({len(training_data)} rows)")

        if len(training_data) < 48:  # Need at least 2 days for Prophet
            print(f"Warning: Insufficient training data ({len(training_data)} rows). "
                  f"Prophet requires at least 2 days.")
            return None

        # Prepare data for Prophet with feature selection
        prophet_data = self._prepare_prophet_df(training_data, for_training=True)

        # Configure and instantiate Prophet model
        model = self._configure_prophet_model()

        # Add holidays if available
        if holiday_events is not None and not holiday_events.empty:
            print(f"Adding {len(holiday_events)} holiday events to the model.")
            model.holidays = holiday_events

        # Fit the model
        print(f"\nFitting Prophet model on {len(prophet_data)} data points...")
        try:
            start_time_fit = datetime.now()
            model.fit(prophet_data)
            fit_duration = (datetime.now() - start_time_fit).total_seconds()
            print(f"Model fitting completed in {fit_duration:.2f} seconds.")

            # Perform cross-validation to check for overfitting
            cv_metrics = self._perform_cross_validation(model, prophet_data)
            
            # Calculate training metrics
            train_forecast = model.predict(prophet_data)
            y_true = prophet_data['y'].values
            y_pred = train_forecast['yhat'].values
            
            # Pass model floor/cap to clipping in metrics calculation
            self._calculate_and_print_metrics(y_true, y_pred, "Training", self.market_floor, self.market_cap)

            # Store the trained model
            self.model = model

            # Save model and parameters
            self._save_model_and_params()

            return model

        except Exception as e:
            print(f"Error training Prophet model: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _save_model_and_params(self):
        """Save the Prophet model and feature engineering parameters."""
        model_dir = self.config.get('model_path', './models/')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_file = os.path.join(model_dir, 'prophet_model.json')
        params_file = os.path.join(model_dir, 'prophet_params.pkl')

        # Ensure model exists before saving
        if self.model is None:
            print("Error: Cannot save model, it has not been trained successfully.")
            return

        try:
            with open(model_file, 'w') as f:
                f.write(model_to_json(self.model))
            print(f"Saved model to {model_file}")

            params_to_save = {
                'feature_engineering_params': self.feature_engineering_params,
                'regressors_list': self.regressors_list,
                'outlier_handler': getattr(self, 'outlier_handler', None),
                'market_floor': self.market_floor,
                'market_cap': self.market_cap,
                'target_variable': self.target_variable,
                'feature_importances': self.feature_importances,
                'cv_metrics': self.cv_metrics
            }
            joblib.dump(params_to_save, params_file)
            print(f"Saved feature parameters to {params_file}")

        except Exception as e:
            print(f"Error saving model or parameters: {e}")

    def load_model(self):
        """Load a previously trained Prophet model and parameters."""
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is required for this forecaster. Install with: pip install prophet")

        model_dir = self.config.get('model_path', './models/')
        model_file = os.path.join(model_dir, 'prophet_model.json')
        params_file = os.path.join(model_dir, 'prophet_params.pkl')

        if os.path.exists(model_file) and os.path.exists(params_file):
            try:
                with open(model_file, 'r') as f:
                    self.model = model_from_json(f.read())
                print(f"Loaded model from {model_file}")

                params_loaded = joblib.load(params_file)
                self.feature_engineering_params = params_loaded.get('feature_engineering_params', {})
                self.regressors_list = params_loaded.get('regressors_list', [])
                self.outlier_handler = params_loaded.get('outlier_handler', None)
                self.market_floor = params_loaded.get('market_floor', self.market_floor)
                self.market_cap = params_loaded.get('market_cap', self.market_cap)
                self.target_variable = params_loaded.get('target_variable', self.target_variable)
                self.feature_importances = params_loaded.get('feature_importances', {})
                self.cv_metrics = params_loaded.get('cv_metrics', None)

                print(f"Loaded feature parameters from {params_file}")
                print(f" - Using {len(self.regressors_list)} regressors")
                print(f" - Target Variable: {self.target_variable}")
                print(f" - Market Floor/Cap: {self.market_floor}/{self.market_cap}")
                
                if not self.outlier_handler:
                    print("Warning: No outlier handler found in loaded parameters.")

                # Verify model object seems valid after loading
                if not hasattr(self.model, 'predict'):
                    print("Error: Loaded object does not appear to be a valid Prophet model.")
                    self.model = None
                    return None

                return self.model
            except Exception as e:
                print(f"Error loading Prophet model or parameters: {e}")
                self.model = None
                return None
        else:
            print(f"Model or parameters file not found in {model_dir}.")
            return None

    def forecast(self, data_file, start_date, horizon_days):
        """
        Generate price forecasts using the trained Prophet model.

        Args:
            data_file: Path to the data file (used for getting future regressors)
            start_date: Start date (str or datetime) for forecasting
            horizon_days: Number of days to forecast ahead

        Returns:
            DataFrame with hourly price forecasts ('Date', 'Forecast_Price')
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is required for this forecaster. Install with: pip install prophet")

        # Load model or train if needed
        if self.model is None:
            should_retrain = self.config.get('retrain', True)
            loaded_ok = False
            if not should_retrain:
                loaded_model = self.load_model()
                if loaded_model:
                    loaded_ok = True
                else:
                    print("Model loading failed, will attempt retraining.")
                    
            # If loading failed or retraining is enabled
            if not loaded_ok or should_retrain:
                print("Training model before forecasting...")
                train_data_file = data_file or self.config.get('market_input_file')
                if not train_data_file:
                    raise ValueError("Data file needed for training is not specified.")
                trained_model = self.train_model(train_data_file, start_date)
                if not trained_model:
                    raise ValueError("Model training failed. Cannot proceed with forecasting.")

        if self.model is None:
            raise ValueError("Model is not available. Ensure training or loading was successful.")

        # Load data for the future period to get regressors
        full_data = self.load_data(data_file)
        if full_data.empty:
            raise ValueError(f"Failed to load data from {data_file} for forecasting.")
            
        # Preprocess data (use fit_transform_outliers=False to only apply existing handler)
        processed_data, _ = self.preprocess_data(full_data, fit_transform_outliers=False)

        # Create future DataFrame with dates
        start_dt = pd.to_datetime(start_date)
        future_dates = pd.date_range(start=start_dt, periods=horizon_days * 24, freq='H')
        
        # Extract exactly the data we need for the forecast period
        future_data_mask = processed_data['Date'].isin(future_dates)
        future_data = processed_data.loc[future_data_mask].copy()
        
        # Verify we have data for all required dates
        if len(future_data) != len(future_dates):
            missing_dates = set(future_dates) - set(processed_data['Date'])
            raise ValueError(f"Missing data for {len(missing_dates)} forecast dates. Cannot proceed without complete data.")
        
        # Ensure future_data is sorted by date
        future_data = future_data.sort_values('Date').reset_index(drop=True)
        
        # Create the Prophet DataFrame with proper columns
        future_prophet = pd.DataFrame({'ds': future_data['Date']})
        future_prophet['floor'] = self.market_floor
        future_prophet['cap'] = self.market_cap
        
        # Process each regressor that was used during training
        for regressor in self.regressors_list:
            if regressor.endswith('_normalized'):
                # This is a normalized feature, we need to extract and normalize the base feature
                base_feature = regressor.replace('_normalized', '')
                
                # Check if the base feature exists
                if base_feature not in future_data.columns:
                    raise ValueError(f"Required base feature '{base_feature}' not found in forecast data.")
                
                # Get normalization parameters
                feature_mean = self.feature_engineering_params.get(f'{base_feature}_mean')
                feature_std = self.feature_engineering_params.get(f'{base_feature}_std')
                
                if feature_mean is None or feature_std is None:
                    raise ValueError(f"Missing normalization parameters for '{base_feature}'.")
                
                if feature_std <= 1e-6:
                    raise ValueError(f"Standard deviation for '{base_feature}' is too small for normalization.")
                
                # Normalize the feature
                future_prophet[regressor] = (future_data[base_feature] - feature_mean) / feature_std
            else:
                # Regular feature - should exist in the data
                if regressor not in future_data.columns:
                    raise ValueError(f"Required regressor '{regressor}' not found in forecast data.")
                
                # Check for NaN values
                if future_data[regressor].isnull().any():
                    raise ValueError(f"Feature '{regressor}' contains NaN values in forecast period.")
                
                # Add the feature to the Prophet DataFrame
                future_prophet[regressor] = future_data[regressor]
        
        # Validate that all required columns are present
        missing_columns = set(self.regressors_list) - set(future_prophet.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns in forecast data: {missing_columns}")
            
        # Verify no NaN values exist in the dataframe
        if future_prophet.isnull().any().any():
            columns_with_nans = [col for col in future_prophet.columns if future_prophet[col].isnull().any()]
            raise ValueError(f"NaN values found in columns: {columns_with_nans}")

        # Generate forecast
        print(f"Generating Prophet forecast for {len(future_prophet)} hours...")
        try:
            forecast_object = self.model.predict(future_prophet)
        except Exception as e:
            print(f"Error during Prophet predict: {e}")
            raise

        # Create the final forecast DataFrame
        result_df = pd.DataFrame({
            'Date': forecast_object['ds'],
            'Forecast_Price': forecast_object['yhat'].values
        })

        # Final clipping to ensure forecast respects market bounds
        result_df['Forecast_Price'] = result_df['Forecast_Price'].clip(self.market_floor, self.market_cap)

        print(f"Generated {len(result_df)} hourly price forecasts.")
        return result_df

    def _calculate_metrics_dict(self, y_true, y_pred, floor=-np.inf, cap=np.inf):
        """
        Calculate standard error metrics with robust handling for zero/negative prices.
        
        Args:
            y_true: Array of actual values
            y_pred: Array of predicted values
            floor: Lower bound for predictions
            cap: Upper bound for predictions
            
        Returns:
            Dictionary with various error metrics
        """
        metrics = {}
        # Ensure inputs are numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if len(y_true) == 0 or len(y_pred) == 0:
            print("Metrics calculation skipped - empty data.")
            return metrics

        # Clip predictions based on provided floor/cap
        y_pred_clipped = np.clip(y_pred, floor, cap)

        # Standard metrics
        metrics['mae'] = mean_absolute_error(y_true, y_pred_clipped)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred_clipped))
        
        try:
            metrics['r2'] = r2_score(y_true, y_pred_clipped)
        except ValueError:
            metrics['r2'] = np.nan

        # Modified MAPE (capping individual errors at 100%)
        with np.errstate(divide='ignore', invalid='ignore'):
            mod_mape_ind = np.abs(y_pred_clipped - y_true) / np.abs(y_true)
        mod_mape_ind[~np.isfinite(mod_mape_ind)] = 1.0  # Replace inf/nan with 100% error
        mod_mape_ind = np.minimum(mod_mape_ind, 1.0)    # Cap error at 100%
        metrics['mape'] = np.mean(mod_mape_ind) * 100

        # Raw MAPE for reference (only on values > 1)
        mask_raw = np.abs(y_true) > 1.0
        if np.sum(mask_raw) > 0:
            metrics['raw_mape'] = mean_absolute_percentage_error(
                y_true[mask_raw], y_pred_clipped[mask_raw]) * 100
        else:
            metrics['raw_mape'] = np.nan

        # SMAPE (Symmetric MAPE)
        abs_diff = np.abs(y_pred_clipped - y_true)
        abs_sum_half = (np.abs(y_true) + np.abs(y_pred_clipped)) / 2.0
        # Avoid division by zero
        smape_values = np.divide(
            abs_diff, abs_sum_half, 
            out=np.zeros_like(abs_diff), 
            where=abs_sum_half > 1e-9
        )
        metrics['smape'] = np.mean(smape_values) * 100

        # Weighted MAPE
        weights = np.abs(y_true)
        total_weight = np.sum(weights)
        metrics['wmape'] = (np.sum(abs_diff * weights) / total_weight * 100) if total_weight > 1e-9 else np.nan

        # Median Absolute Error
        metrics['medae'] = np.median(abs_diff)

        return metrics

    def _calculate_and_print_metrics(self, y_true, y_pred, label="", floor=-np.inf, cap=np.inf):
        """Calculate and print error metrics with meaningful labels."""
        metrics = self._calculate_metrics_dict(y_true, y_pred, floor, cap)

        print(f"\n{label} Metrics:")
        if not metrics:
            print("  No metrics calculated.")
            return

        print(f"  MAE:   {metrics.get('mae', np.nan):.2f} €/MWh")
        print(f"  RMSE:  {metrics.get('rmse', np.nan):.2f} €/MWh")
        print(f"  MedAE: {metrics.get('medae', np.nan):.2f} €/MWh")
        print(f"  MAPE:  {metrics.get('mape', np.nan):.2f}% (Modified, capped at 100% ind. error)")
        print(f"  Raw MAPE: {metrics.get('raw_mape', np.nan):.2f}% (on abs(actual) > 1)")
        print(f"  SMAPE: {metrics.get('smape', np.nan):.2f}%")
        print(f"  WMAPE: {metrics.get('wmape', np.nan):.2f}% (Weighted by abs(actual))")
        print(f"  R²:    {metrics.get('r2', np.nan):.4f}")
        print("-" * (len(label) + 10))


# testing during develpemnet 
if __name__ == '__main__':
    # Example configuration (would be loaded from config.yaml in practice)
    config = {
        'target_variable': 'Price',
        'training_days': 90,
        'model_path': './models/',
        'retrain': True,
        'market_input_file': 'scaled_data_NL4.xlsx',
        'feature_sets': ['hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday', 'load', 'solar_wind'],
        'prophet': {
            'changepoint_prior_scale': 0.1,
            'seasonality_prior_scale': 10.0,
            'holidays_prior_scale': 10.0,
            'daily_seasonality': 5.0,
            'weekly_seasonality': 10.0,
            'yearly_seasonality': 'auto',
            'seasonality_mode': 'multiplicative',
            'country_code': 'NL',
            'floor': -500.0,
            'cap': 4000.0
        },
        'cv': {
            'enabled': True,
            'initial': '30 days',
            'period': '7 days',
            'horizon': '7 days'
        },
        'feature_selection': {
            'enabled': True,
            'max_features': 10,
            'correlation_threshold': 0.85
        },
        'outlier_handling': {
            'enabled': True,
            'method': 'clip',
            'threshold': 3
        }
    }

    # Initialize forecaster
    forecaster = ElectricityPriceForecaster(config)

    try:
        # Generate a forecast
        data_file = config['market_input_file']
        start_date = '2024-06-01'
        horizon_days = 5

        forecast_df = forecaster.forecast(data_file, start_date, horizon_days)

        # Plot the forecast
        actual_data = forecaster.load_data(data_file)
        forecaster.plot_forecast(
            forecast_df, 
            actual_data, 
            output_dir='./plots', 
            filename_prefix=f"forecast_{start_date}"
        )

    except Exception as e:
        print(f"\nError during example usage: {e}")
        import traceback
        traceback.print_exc()
        print("\nEnsure Prophet is installed ('pip install prophet') and data file exists.")