"""Time series forecasting models for market trend analysis."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from typing import Tuple, Optional, Dict, Any
import logging

# ARIMA and statistical models
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.stats.diagnostic import acorr_ljungbox
except ImportError:
    print("Warning: statsmodels not installed. ARIMA models will be unavailable.")
    ARIMA = None

# Prophet
try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
except ImportError:
    print("Warning: prophet not installed. Prophet models will be unavailable.")
    Prophet = None

# MLflow integration
try:
    import mlflow
    import mlflow.sklearn
except ImportError:
    mlflow = None

from src.utils.config import config

logger = logging.getLogger(__name__)

class ARIMAForecaster:
    """ARIMA-based time series forecasting model."""
    
    def __init__(self, order: Tuple[int, int, int] = None):
        """Initialize ARIMA forecaster.
        
        Args:
            order: ARIMA order (p, d, q). If None, will be auto-determined.
        """
        if ARIMA is None:
            raise ImportError("statsmodels is required for ARIMA models")
        
        self.order = order or config.get('models.time_series.arima.order', (1, 1, 1))
        self.model = None
        self.fitted_model = None
        self.training_data = None
    
    def check_stationarity(self, ts: pd.Series, significance_level: float = 0.05) -> bool:
        """Check if time series is stationary using Augmented Dickey-Fuller test.
        
        Args:
            ts: Time series data
            significance_level: Significance level for the test
            
        Returns:
            True if series is stationary, False otherwise
        """
        result = adfuller(ts.dropna())
        p_value = result[1]
        
        logger.info(f"ADF Test p-value: {p_value:.6f}")
        
        if p_value <= significance_level:
            logger.info("Time series is stationary")
            return True
        else:
            logger.info("Time series is not stationary")
            return False
    
    def auto_arima_order(self, ts: pd.Series, max_p: int = 5, max_d: int = 2, max_q: int = 5) -> Tuple[int, int, int]:
        """Automatically determine optimal ARIMA order using grid search.
        
        Args:
            ts: Time series data
            max_p: Maximum p parameter
            max_d: Maximum d parameter
            max_q: Maximum q parameter
            
        Returns:
            Optimal (p, d, q) order
        """
        best_aic = float('inf')
        best_order = (0, 0, 0)
        
        logger.info("Searching for optimal ARIMA parameters...")
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(ts, order=(p, d, q))
                        fitted = model.fit()
                        
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                            
                    except Exception as e:
                        continue
        
        logger.info(f"Optimal ARIMA order: {best_order} with AIC: {best_aic:.4f}")
        return best_order
    
    def fit(self, ts: pd.Series, auto_order: bool = False) -> 'ARIMAForecaster':
        """Fit ARIMA model to time series data.
        
        Args:
            ts: Time series data with datetime index
            auto_order: Whether to automatically determine ARIMA order
            
        Returns:
            Self for method chaining
        """
        self.training_data = ts.copy()
        
        if auto_order:
            self.order = self.auto_arima_order(ts)
        
        logger.info(f"Fitting ARIMA{self.order} model...")
        
        try:
            self.model = ARIMA(ts, order=self.order)
            self.fitted_model = self.model.fit()
            
            logger.info("ARIMA model fitted successfully")
            logger.info(f"Model AIC: {self.fitted_model.aic:.4f}")
            
            # Log model summary
            logger.info(f"Model Summary:\n{self.fitted_model.summary()}")
            
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {e}")
            raise
        
        return self
    
    def forecast(self, steps: int = 30) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Generate forecasts.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Tuple of (forecasts, lower_bound, upper_bound)
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        forecast_result = self.fitted_model.forecast(steps=steps, alpha=0.05)
        forecast_values = forecast_result[0]
        conf_int = forecast_result[1]
        
        # Create forecast index
        last_date = self.training_data.index[-1]
        forecast_index = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=steps,
            freq='D'
        )
        
        forecast_series = pd.Series(forecast_values, index=forecast_index, name='forecast')
        lower_bound = pd.Series(conf_int[:, 0], index=forecast_index, name='lower_bound')
        upper_bound = pd.Series(conf_int[:, 1], index=forecast_index, name='upper_bound')
        
        return forecast_series, lower_bound, upper_bound
    
    def evaluate(self, test_data: pd.Series) -> Dict[str, float]:
        """Evaluate model performance on test data.
        
        Args:
            test_data: Test time series data
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before evaluation")
        
        forecast_steps = len(test_data)
        forecasts, _, _ = self.forecast(forecast_steps)
        
        # Calculate metrics
        mae = np.mean(np.abs(forecasts - test_data))
        mse = np.mean((forecasts - test_data) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((test_data - forecasts) / test_data)) * 100
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape
        }
        
        logger.info(f"ARIMA Model Evaluation Metrics: {metrics}")
        return metrics

class ProphetForecaster:
    """Facebook Prophet-based time series forecasting model."""
    
    def __init__(self, **prophet_params):
        """Initialize Prophet forecaster.
        
        Args:
            **prophet_params: Parameters to pass to Prophet model
        """
        if Prophet is None:
            raise ImportError("prophet is required for Prophet models")
        
        # Default Prophet parameters
        default_params = config.get('models.time_series.prophet', {})
        self.prophet_params = {**default_params, **prophet_params}
        
        self.model = None
        self.training_data = None
    
    def fit(self, ts: pd.Series) -> 'ProphetForecaster':
        """Fit Prophet model to time series data.
        
        Args:
            ts: Time series data with datetime index
            
        Returns:
            Self for method chaining
        """
        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        df = ts.reset_index()
        df.columns = ['ds', 'y']
        self.training_data = df.copy()
        
        logger.info("Fitting Prophet model...")
        
        try:
            self.model = Prophet(**self.prophet_params)
            self.model.fit(df)
            
            logger.info("Prophet model fitted successfully")
            
        except Exception as e:
            logger.error(f"Error fitting Prophet model: {e}")
            raise
        
        return self
    
    def forecast(self, steps: int = 30) -> pd.DataFrame:
        """Generate forecasts using Prophet.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            DataFrame with forecasts and confidence intervals
        """
        if self.model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=steps)
        
        # Generate forecasts
        forecast = self.model.predict(future)
        
        # Return only the forecast period
        return forecast.tail(steps)
    
    def cross_validate_model(self, initial: str = '365 days', 
                           period: str = '90 days', 
                           horizon: str = '30 days') -> pd.DataFrame:
        """Perform cross-validation on the Prophet model.
        
        Args:
            initial: Initial training period
            period: Period between cutoffs
            horizon: Forecast horizon
            
        Returns:
            Cross-validation results
        """
        if self.model is None:
            raise ValueError("Model must be fitted before cross-validation")
        
        if cross_validation is None:
            logger.warning("Prophet diagnostics not available")
            return pd.DataFrame()
        
        logger.info("Performing Prophet cross-validation...")
        
        try:
            cv_results = cross_validation(
                self.model, 
                initial=initial, 
                period=period, 
                horizon=horizon
            )
            
            # Calculate performance metrics
            metrics = performance_metrics(cv_results)
            logger.info(f"Prophet CV Metrics:\n{metrics}")
            
            return cv_results
            
        except Exception as e:
            logger.error(f"Error during cross-validation: {e}")
            return pd.DataFrame()
    
    def plot_forecast(self, forecast: pd.DataFrame, figsize: Tuple[int, int] = (12, 8)):
        """Plot Prophet forecast.
        
        Args:
            forecast: Forecast DataFrame from Prophet
            figsize: Figure size
        """
        if self.model is None:
            raise ValueError("Model must be fitted before plotting")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Plot forecast
        self.model.plot(forecast, ax=ax1)
        ax1.set_title('Prophet Forecast')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value')
        
        # Plot components
        self.model.plot_components(forecast, ax=ax2)
        
        plt.tight_layout()
        plt.show()

def prepare_time_series_data(df: pd.DataFrame, 
                           date_column: str, 
                           value_column: str,
                           freq: str = 'D') -> pd.Series:
    """Prepare time series data for modeling.
    
    Args:
        df: Input DataFrame
        date_column: Name of the date column
        value_column: Name of the value column
        freq: Frequency for resampling
        
    Returns:
        Time series with proper datetime index
    """
    # Ensure date column is datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Set date column as index
    ts = df.set_index(date_column)[value_column]
    
    # Sort by date
    ts = ts.sort_index()
    
    # Resample to specified frequency and forward fill missing values
    ts = ts.resample(freq).mean().fillna(method='ffill')
    
    return ts

def train_time_series_models(df: pd.DataFrame, 
                           date_column: str = 'lease_start_date',
                           value_column: str = 'monthly_rent',
                           test_size: float = 0.2) -> Dict[str, Any]:
    """Train and evaluate time series models.
    
    Args:
        df: Input DataFrame
        date_column: Date column name
        value_column: Target value column name
        test_size: Proportion of data for testing
        
    Returns:
        Dictionary containing trained models and results
    """
    logger.info(f"Training time series models for {value_column}")
    
    # Prepare time series data
    ts = prepare_time_series_data(df, date_column, value_column)
    
    # Split data
    split_idx = int(len(ts) * (1 - test_size))
    train_ts = ts[:split_idx]
    test_ts = ts[split_idx:]
    
    results = {
        'data': {
            'train': train_ts,
            'test': test_ts,
            'full': ts
        },
        'models': {},
        'forecasts': {},
        'metrics': {}
    }
    
    # Train ARIMA model
    if ARIMA is not None:
        try:
            logger.info("Training ARIMA model...")
            arima_model = ARIMAForecaster()
            arima_model.fit(train_ts, auto_order=True)
            
            # Generate forecasts
            arima_forecast, arima_lower, arima_upper = arima_model.forecast(len(test_ts))
            
            # Evaluate
            arima_metrics = arima_model.evaluate(test_ts)
            
            results['models']['arima'] = arima_model
            results['forecasts']['arima'] = {
                'forecast': arima_forecast,
                'lower_bound': arima_lower,
                'upper_bound': arima_upper
            }
            results['metrics']['arima'] = arima_metrics
            
        except Exception as e:
            logger.error(f"Error training ARIMA model: {e}")
    
    # Train Prophet model
    if Prophet is not None:
        try:
            logger.info("Training Prophet model...")
            prophet_model = ProphetForecaster()
            prophet_model.fit(train_ts)
            
            # Generate forecasts
            prophet_forecast = prophet_model.forecast(len(test_ts))
            
            # Calculate metrics for Prophet
            prophet_predictions = prophet_forecast['yhat'].values
            prophet_mae = np.mean(np.abs(prophet_predictions - test_ts.values))
            prophet_mse = np.mean((prophet_predictions - test_ts.values) ** 2)
            prophet_rmse = np.sqrt(prophet_mse)
            prophet_mape = np.mean(np.abs((test_ts.values - prophet_predictions) / test_ts.values)) * 100
            
            prophet_metrics = {
                'mae': prophet_mae,
                'mse': prophet_mse,
                'rmse': prophet_rmse,
                'mape': prophet_mape
            }
            
            results['models']['prophet'] = prophet_model
            results['forecasts']['prophet'] = prophet_forecast
            results['metrics']['prophet'] = prophet_metrics
            
        except Exception as e:
            logger.error(f"Error training Prophet model: {e}")
    
    return results