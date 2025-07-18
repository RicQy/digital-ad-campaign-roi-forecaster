# Forecasting Module

The forecasting module provides comprehensive forecasting capabilities for campaign conversion predictions. It includes baseline models, time series models, and model selection functionality.

## Components

### 1. Baseline Forecaster (`baseline.py`)
- **BaselineForecaster**: Uses OLS/linear regression to predict conversions based on spend and campaign features
- Features: spend, impressions, clicks, temporal features, and categorical encoding
- Supports confidence intervals and feature importance analysis

### 2. Time Series Forecaster (`time_series.py`)
- **SeasonalNaiveForecaster**: Predicts based on seasonal patterns (e.g., same day of week)
- **ARIMAForecaster**: Uses ARIMA models for time series forecasting (requires statsmodels)
- **TimeSeriesForecaster**: Unified interface for time series methods

### 3. Model Selector (`model_selector.py`)
- **ModelSelector**: Compares multiple models using cross-validation
- **EnsembleForecaster**: Combines multiple models with weighted averages
- Automatic model selection based on performance metrics

## Usage Examples

### Basic Baseline Forecasting
```python
from ad_roi_forecaster.forecasting import BaselineForecaster

# Create and train model
forecaster = BaselineForecaster()
forecaster.fit(df)

# Make predictions
predictions = forecaster.predict(df)

# Future forecasting
future_forecasts = forecaster.forecast_future(df, future_periods=30, spend_scenario=200.0)
```

### Time Series Forecasting
```python
from ad_roi_forecaster.forecasting import TimeSeriesForecaster

# Seasonal naive forecasting
ts_forecaster = TimeSeriesForecaster(method='seasonal_naive')
ts_forecaster.fit(df)
predictions = ts_forecaster.predict(periods=30)

# ARIMA forecasting (if statsmodels available)
arima_forecaster = TimeSeriesForecaster(method='arima')
arima_forecaster.fit(df)
predictions = arima_forecaster.predict(periods=30)
```

### Model Selection
```python
from ad_roi_forecaster.forecasting import ModelSelector

# Compare multiple models
selector = ModelSelector(cv_folds=5, scoring='mae')
selector.fit(df)

# Get best model
best_model = selector.best_model
model_info = selector.get_best_model_info()

# Model comparison
comparison = selector.get_model_comparison()
```

### Convenience Functions
```python
from ad_roi_forecaster.forecasting import (
    forecast_conversions_baseline,
    forecast_conversions_time_series,
    forecast_with_best_model
)

# Quick baseline forecasting
baseline_forecasts = forecast_conversions_baseline(df, future_periods=30)

# Quick time series forecasting
ts_forecasts = forecast_conversions_time_series(df, future_periods=30, method='seasonal_naive')

# End-to-end forecasting with model selection
forecasts, model_info = forecast_with_best_model(df, future_periods=30, spend_scenario=250.0)
```

## Output Format

All forecasting methods return a pandas DataFrame with the following columns:
- `predicted_conversions`: Forecasted conversion values
- `lower_bound`: Lower bound of confidence interval
- `upper_bound`: Upper bound of confidence interval
- `confidence_level`: Confidence level (default 0.95)
- `date`: Date for each forecast (when applicable)
- `model_used`: Name of the model used for forecasting

## Model Performance

The module includes comprehensive evaluation metrics:
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of Determination

## Cross-Validation

Time series cross-validation is used to evaluate models while respecting temporal order:
- Training data: Earlier time periods
- Test data: Later time periods
- Multiple folds with expanding windows

## Dependencies

**Required:**
- pandas
- numpy
- scikit-learn
- scipy

**Optional:**
- statsmodels (for ARIMA functionality)

## Best Practices

1. **Data Preparation**: Ensure your data has `date`, `spend`, `impressions`, `clicks`, and `conversions` columns
2. **Model Selection**: Use cross-validation to select the best model for your data
3. **Confidence Intervals**: Always consider prediction uncertainty
4. **Validation**: Evaluate models on out-of-sample data before deployment
5. **Seasonal Patterns**: For time series with weekly patterns, use seasonal_period=7

## Error Handling

The module includes robust error handling for:
- Insufficient data
- Missing features
- Model fitting failures
- Prediction errors

## Performance Considerations

- **Baseline models**: Fast training and prediction, good for real-time applications
- **Time series models**: Better for capturing temporal patterns, slower for large datasets
- **Model selection**: Computationally intensive but provides best results
- **Ensemble methods**: Highest accuracy but longest training time
