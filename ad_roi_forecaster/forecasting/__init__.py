"""
Forecasting module for campaign conversion predictions.

This module provides various forecasting models and utilities for predicting
campaign conversions, including baseline models, time series models, and
model selection capabilities.

Main components:
- BaselineForecaster: Linear regression based forecasting
- TimeSeriesForecaster: Time series based forecasting (seasonal naive, ARIMA)
- ModelSelector: Cross-validation and model selection
- EnsembleForecaster: Ensemble forecasting combining multiple models

Convenience functions:
- forecast_conversions_baseline: Quick baseline forecasting
- forecast_conversions_time_series: Quick time series forecasting
- select_best_model: Automatic model selection
- compare_models: Compare multiple models
- forecast_with_best_model: End-to-end forecasting with model selection
"""

# Import main classes
from .baseline import (
    BaselineForecaster,
    create_baseline_forecaster,
    forecast_conversions_baseline,
)

from .time_series import (
    TimeSeriesForecaster,
    SeasonalNaiveForecaster,
    ARIMAForecaster,
    create_time_series_forecaster,
    forecast_conversions_time_series,
    decompose_time_series,
)

from .model_selector import (
    ModelSelector,
    EnsembleForecaster,
    select_best_model,
    compare_models,
    forecast_with_best_model,
)

# Export all main components
__all__ = [
    # Main forecasting classes
    "BaselineForecaster",
    "TimeSeriesForecaster",
    "SeasonalNaiveForecaster",
    "ARIMAForecaster",
    "ModelSelector",
    "EnsembleForecaster",
    # Convenience functions
    "forecast_conversions_baseline",
    "forecast_conversions_time_series",
    "select_best_model",
    "compare_models",
    "forecast_with_best_model",
    # Utility functions
    "create_baseline_forecaster",
    "create_time_series_forecaster",
    "decompose_time_series",
]
