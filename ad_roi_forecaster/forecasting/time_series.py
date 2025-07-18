"""
Time series forecasting module for campaign conversions.

This module implements time series forecasting models including
seasonal naive forecasting and ARIMA models for predicting
daily conversions based on historical patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
from scipy import stats
import warnings

# Handle statsmodels import gracefully
try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.stats.diagnostic import acorr_ljungbox

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. ARIMA functionality will be limited.")

logger = logging.getLogger(__name__)


class SeasonalNaiveForecaster:
    """
    Seasonal naive forecasting model.

    This model predicts future values based on the same period
    from the previous season (e.g., same day of week, same month).
    """

    def __init__(self, seasonal_period: int = 7, confidence_level: float = 0.95):
        """
        Initialize the seasonal naive forecaster.

        Args:
            seasonal_period: Number of periods in a season (e.g., 7 for weekly).
            confidence_level: Confidence level for prediction intervals.
        """
        self.seasonal_period = seasonal_period
        self.confidence_level = confidence_level
        self.is_trained = False
        self.historical_data = None
        self.seasonal_errors = None

    def fit(self, df: pd.DataFrame) -> "SeasonalNaiveForecaster":
        """
        Fit the seasonal naive model to the training data.

        Args:
            df: Training dataframe with date and conversions columns.

        Returns:
            Self for method chaining.
        """
        logger.info("Training seasonal naive forecasting model...")

        # Prepare data
        df_copy = df.copy()
        df_copy["date"] = pd.to_datetime(df_copy["date"])
        df_copy = df_copy.sort_values("date")

        # Aggregate by date to get daily conversions
        daily_conversions = df_copy.groupby("date")["conversions"].sum().reset_index()
        daily_conversions = daily_conversions.sort_values("date")

        # Store historical data
        self.historical_data = daily_conversions

        # Calculate seasonal errors for confidence intervals
        self._calculate_seasonal_errors()

        self.is_trained = True
        logger.info("Seasonal naive model trained successfully.")

        return self

    def _calculate_seasonal_errors(self):
        """Calculate seasonal forecast errors for confidence intervals."""
        if len(self.historical_data) <= self.seasonal_period:
            # Not enough data for seasonal patterns
            self.seasonal_errors = np.std(self.historical_data["conversions"])
            return

        # Calculate errors for each seasonal period
        errors = []
        data = self.historical_data["conversions"].values

        for i in range(self.seasonal_period, len(data)):
            seasonal_index = i % self.seasonal_period
            # Find the previous occurrence of the same seasonal period
            prev_value = data[i - self.seasonal_period]
            actual_value = data[i]
            error = actual_value - prev_value
            errors.append(error)

        if errors:
            self.seasonal_errors = np.std(errors)
        else:
            self.seasonal_errors = np.std(data)

    def predict(self, periods: int) -> pd.DataFrame:
        """
        Make predictions for future periods.

        Args:
            periods: Number of future periods to predict.

        Returns:
            DataFrame with predictions and confidence intervals.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Get the last date from historical data
        last_date = self.historical_data["date"].max()

        # Create future dates
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1), periods=periods, freq="D"
        )

        # Make predictions
        predictions = []
        conversions = self.historical_data["conversions"].values

        for i, date in enumerate(future_dates):
            # Calculate seasonal index
            total_periods = len(conversions) + i
            seasonal_index = total_periods % self.seasonal_period

            # Find the most recent value for this seasonal period
            if len(conversions) >= self.seasonal_period:
                # Use the value from the same seasonal period
                seasonal_offset = seasonal_index - (
                    len(conversions) % self.seasonal_period
                )
                if seasonal_offset <= 0:
                    seasonal_offset += self.seasonal_period

                reference_index = len(conversions) - seasonal_offset
                if reference_index >= 0:
                    prediction = conversions[reference_index]
                else:
                    prediction = conversions[seasonal_index % len(conversions)]
            else:
                # Not enough data for full seasonal pattern
                prediction = conversions[seasonal_index % len(conversions)]

            predictions.append(prediction)

        # Calculate confidence intervals
        alpha = 1 - self.confidence_level
        z_score = stats.norm.ppf(1 - alpha / 2)
        margin_of_error = z_score * self.seasonal_errors

        # Create results dataframe
        results = pd.DataFrame(
            {
                "date": future_dates,
                "predicted_conversions": predictions,
                "lower_bound": np.maximum(0, np.array(predictions) - margin_of_error),
                "upper_bound": np.array(predictions) + margin_of_error,
                "prediction_std": self.seasonal_errors,
                "confidence_level": self.confidence_level,
            }
        )

        return results

    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the model using backtesting.

        Args:
            df: Test dataframe with date and conversions columns.

        Returns:
            Dictionary with evaluation metrics.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        # Prepare test data
        df_copy = df.copy()
        df_copy["date"] = pd.to_datetime(df_copy["date"])
        daily_conversions = df_copy.groupby("date")["conversions"].sum().reset_index()
        daily_conversions = daily_conversions.sort_values("date")

        # Make predictions for test period
        predictions = self.predict(len(daily_conversions))

        # Calculate metrics
        y_true = daily_conversions["conversions"].values
        y_pred = predictions["predicted_conversions"].values[: len(y_true)]

        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100

        return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape}


class ARIMAForecaster:
    """
    ARIMA forecasting model for time series data.

    This model uses ARIMA (AutoRegressive Integrated Moving Average)
    to forecast conversions based on historical patterns.
    """

    def __init__(
        self,
        order: Optional[Tuple[int, int, int]] = None,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        confidence_level: float = 0.95,
    ):
        """
        Initialize the ARIMA forecaster.

        Args:
            order: ARIMA order (p, d, q). If None, will be auto-selected.
            seasonal_order: Seasonal ARIMA order (P, D, Q, S). If None, no seasonality.
            confidence_level: Confidence level for prediction intervals.
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for ARIMA functionality")

        self.order = order
        self.seasonal_order = seasonal_order
        self.confidence_level = confidence_level
        self.model = None
        self.fitted_model = None
        self.is_trained = False
        self.historical_data = None

    def _check_stationarity(self, timeseries: pd.Series) -> bool:
        """
        Check if the time series is stationary using ADF test.

        Args:
            timeseries: Time series data.

        Returns:
            True if stationary, False otherwise.
        """
        result = adfuller(timeseries.dropna())
        p_value = result[1]
        return p_value <= 0.05

    def _auto_select_order(self, timeseries: pd.Series) -> Tuple[int, int, int]:
        """
        Auto-select ARIMA order using AIC criterion.

        Args:
            timeseries: Time series data.

        Returns:
            Optimal ARIMA order (p, d, q).
        """
        best_aic = float("inf")
        best_order = (0, 0, 0)

        # Test different combinations
        for p in range(0, 4):
            for d in range(0, 3):
                for q in range(0, 4):
                    try:
                        model = ARIMA(timeseries, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except:
                        continue

        return best_order

    def fit(self, df: pd.DataFrame) -> "ARIMAForecaster":
        """
        Fit the ARIMA model to the training data.

        Args:
            df: Training dataframe with date and conversions columns.

        Returns:
            Self for method chaining.
        """
        logger.info("Training ARIMA forecasting model...")

        # Prepare data
        df_copy = df.copy()
        df_copy["date"] = pd.to_datetime(df_copy["date"])
        df_copy = df_copy.sort_values("date")

        # Aggregate by date to get daily conversions
        daily_conversions = df_copy.groupby("date")["conversions"].sum().reset_index()
        daily_conversions = daily_conversions.sort_values("date")

        # Create time series
        ts = pd.Series(
            daily_conversions["conversions"].values, index=daily_conversions["date"]
        )

        # Store historical data
        self.historical_data = ts

        # Auto-select order if not provided
        if self.order is None:
            self.order = self._auto_select_order(ts)
            logger.info(f"Auto-selected ARIMA order: {self.order}")

        # Fit the model
        try:
            self.model = ARIMA(ts, order=self.order, seasonal_order=self.seasonal_order)
            self.fitted_model = self.model.fit()
            self.is_trained = True
            logger.info(
                f"ARIMA model trained successfully. AIC: {self.fitted_model.aic:.2f}"
            )
        except Exception as e:
            logger.error(f"Failed to fit ARIMA model: {e}")
            raise

        return self

    def predict(self, periods: int) -> pd.DataFrame:
        """
        Make predictions for future periods.

        Args:
            periods: Number of future periods to predict.

        Returns:
            DataFrame with predictions and confidence intervals.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Make forecasts
        forecast = self.fitted_model.forecast(steps=periods)
        conf_int = self.fitted_model.get_forecast(steps=periods).conf_int(
            alpha=1 - self.confidence_level
        )

        # Create future dates
        last_date = self.historical_data.index[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1), periods=periods, freq="D"
        )

        # Create results dataframe
        results = pd.DataFrame(
            {
                "date": future_dates,
                "predicted_conversions": np.maximum(0, forecast.values),
                "lower_bound": np.maximum(0, conf_int.iloc[:, 0].values),
                "upper_bound": np.maximum(0, conf_int.iloc[:, 1].values),
                "confidence_level": self.confidence_level,
            }
        )

        return results

    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the model using backtesting.

        Args:
            df: Test dataframe with date and conversions columns.

        Returns:
            Dictionary with evaluation metrics.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        # Prepare test data
        df_copy = df.copy()
        df_copy["date"] = pd.to_datetime(df_copy["date"])
        daily_conversions = df_copy.groupby("date")["conversions"].sum().reset_index()
        daily_conversions = daily_conversions.sort_values("date")

        # Make predictions for test period
        predictions = self.predict(len(daily_conversions))

        # Calculate metrics
        y_true = daily_conversions["conversions"].values
        y_pred = predictions["predicted_conversions"].values[: len(y_true)]

        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100

        # Add model-specific metrics
        metrics = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "mape": mape,
            "aic": self.fitted_model.aic,
            "bic": self.fitted_model.bic,
        }

        return metrics

    def get_model_summary(self) -> str:
        """
        Get a summary of the fitted model.

        Returns:
            String representation of the model summary.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting summary")

        return str(self.fitted_model.summary())


class TimeSeriesForecaster:
    """
    Combined time series forecasting class.

    This class provides a unified interface for different time series
    forecasting methods including seasonal naive and ARIMA.
    """

    def __init__(
        self,
        method: str = "seasonal_naive",
        seasonal_period: int = 7,
        confidence_level: float = 0.95,
        arima_order: Optional[Tuple[int, int, int]] = None,
    ):
        """
        Initialize the time series forecaster.

        Args:
            method: Forecasting method ('seasonal_naive' or 'arima').
            seasonal_period: Seasonal period for seasonal naive method.
            confidence_level: Confidence level for prediction intervals.
            arima_order: ARIMA order for ARIMA method.
        """
        self.method = method
        self.seasonal_period = seasonal_period
        self.confidence_level = confidence_level
        self.arima_order = arima_order

        # Initialize the appropriate forecaster
        if method == "seasonal_naive":
            self.forecaster = SeasonalNaiveForecaster(
                seasonal_period=seasonal_period, confidence_level=confidence_level
            )
        elif method == "arima":
            self.forecaster = ARIMAForecaster(
                order=arima_order, confidence_level=confidence_level
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        self.is_trained = False

    def fit(self, df: pd.DataFrame) -> "TimeSeriesForecaster":
        """
        Fit the time series model to the training data.

        Args:
            df: Training dataframe with date and conversions columns.

        Returns:
            Self for method chaining.
        """
        self.forecaster.fit(df)
        self.is_trained = True
        return self

    def predict(self, periods: int) -> pd.DataFrame:
        """
        Make predictions for future periods.

        Args:
            periods: Number of future periods to predict.

        Returns:
            DataFrame with predictions and confidence intervals.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        predictions = self.forecaster.predict(periods)
        predictions["method"] = self.method

        return predictions

    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the model using backtesting.

        Args:
            df: Test dataframe with date and conversions columns.

        Returns:
            Dictionary with evaluation metrics.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        metrics = self.forecaster.evaluate(df)
        metrics["method"] = self.method

        return metrics


def decompose_time_series(
    df: pd.DataFrame, period: int = 7, model: str = "additive"
) -> Dict[str, pd.DataFrame]:
    """
    Decompose time series into trend, seasonal, and residual components.

    Args:
        df: Dataframe with date and conversions columns.
        period: Seasonal period for decomposition.
        model: Decomposition model ('additive' or 'multiplicative').

    Returns:
        Dictionary with decomposed components.
    """
    # Prepare data
    df_copy = df.copy()
    df_copy["date"] = pd.to_datetime(df_copy["date"])
    daily_conversions = df_copy.groupby("date")["conversions"].sum().reset_index()
    daily_conversions = daily_conversions.sort_values("date")

    # Create time series
    ts = pd.Series(
        daily_conversions["conversions"].values, index=daily_conversions["date"]
    )

    # Perform decomposition
    if STATSMODELS_AVAILABLE and len(ts) >= 2 * period:
        decomposition = seasonal_decompose(ts, period=period, model=model)

        return {
            "original": pd.DataFrame({"date": ts.index, "values": ts.values}),
            "trend": pd.DataFrame(
                {
                    "date": decomposition.trend.index,
                    "values": decomposition.trend.values,
                }
            ),
            "seasonal": pd.DataFrame(
                {
                    "date": decomposition.seasonal.index,
                    "values": decomposition.seasonal.values,
                }
            ),
            "residual": pd.DataFrame(
                {
                    "date": decomposition.resid.index,
                    "values": decomposition.resid.values,
                }
            ),
        }
    else:
        # Simple trend estimation if statsmodels not available
        trend = ts.rolling(window=period, center=True).mean()
        residual = ts - trend

        return {
            "original": pd.DataFrame({"date": ts.index, "values": ts.values}),
            "trend": pd.DataFrame({"date": trend.index, "values": trend.values}),
            "seasonal": pd.DataFrame({"date": ts.index, "values": [0] * len(ts)}),
            "residual": pd.DataFrame(
                {"date": residual.index, "values": residual.values}
            ),
        }


def forecast_conversions_time_series(
    df: pd.DataFrame,
    future_periods: int = 30,
    method: str = "seasonal_naive",
    seasonal_period: int = 7,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """
    Convenience function to create time series forecasts.

    Args:
        df: Historical campaign data.
        future_periods: Number of future periods to forecast.
        method: Forecasting method ('seasonal_naive' or 'arima').
        seasonal_period: Seasonal period for seasonal naive method.
        confidence_level: Confidence level for predictions.

    Returns:
        DataFrame with forecasted conversions and confidence intervals.
    """
    # Create and train model
    forecaster = TimeSeriesForecaster(
        method=method,
        seasonal_period=seasonal_period,
        confidence_level=confidence_level,
    )
    forecaster.fit(df)

    # Make forecasts
    forecasts = forecaster.predict(future_periods)

    return forecasts


def create_time_series_forecaster(
    df: pd.DataFrame, method: str = "seasonal_naive", test_size: float = 0.2
) -> Tuple[TimeSeriesForecaster, Dict[str, float]]:
    """
    Create and train a time series forecasting model.

    Args:
        df: Campaign data for training.
        method: Forecasting method to use.
        test_size: Fraction of data to use for testing.

    Returns:
        Tuple of (trained model, evaluation metrics).
    """
    # Prepare data
    df_copy = df.copy()
    df_copy["date"] = pd.to_datetime(df_copy["date"])
    df_copy = df_copy.sort_values("date")

    # Split data by time (not randomly)
    n_test = int(len(df_copy) * test_size)
    train_df = df_copy.iloc[:-n_test]
    test_df = df_copy.iloc[-n_test:]

    # Create and train model
    forecaster = TimeSeriesForecaster(method=method)
    forecaster.fit(train_df)

    # Evaluate model
    metrics = forecaster.evaluate(test_df)

    return forecaster, metrics
