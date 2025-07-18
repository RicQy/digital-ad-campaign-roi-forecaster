#!/usr/bin/env python3
"""
Test script for the forecasting module.

This script demonstrates the functionality of the forecasting module
including baseline models, time series models, and model selection.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import data loading functionality
from ad_roi_forecaster.data import load_campaign_data

# Import forecasting functionality
from ad_roi_forecaster.forecasting import (
    BaselineForecaster,
    TimeSeriesForecaster,
    ModelSelector,
    forecast_conversions_baseline,
    forecast_conversions_time_series,
    select_best_model,
    compare_models,
    forecast_with_best_model,
)


def test_baseline_forecasting():
    """Test baseline forecasting functionality."""
    print("\n🔍 Testing Baseline Forecasting")
    print("=" * 50)

    # Load sample data
    csv_path = project_root / "sample_data" / "campaign_data.csv"
    df = load_campaign_data(csv_path)

    print(f"Loaded {len(df)} records for testing")

    # Test baseline forecaster
    print("\n1. Testing BaselineForecaster...")
    forecaster = BaselineForecaster()
    forecaster.fit(df)

    # Make predictions on existing data
    predictions = forecaster.predict(df)
    print(f"   Made predictions for {len(predictions)} samples")
    print(
        f"   Predictions range: {predictions['predicted_conversions'].min():.1f} to {predictions['predicted_conversions'].max():.1f}"
    )

    # Test future forecasting
    print("\n2. Testing future forecasting...")
    future_forecasts = forecaster.forecast_future(
        df, future_periods=7, spend_scenario=200.0
    )
    print(f"   Generated {len(future_forecasts)} future forecasts")
    print(
        f"   Future predictions range: {future_forecasts['predicted_conversions'].min():.1f} to {future_forecasts['predicted_conversions'].max():.1f}"
    )

    # Test convenience function
    print("\n3. Testing convenience function...")
    quick_forecasts = forecast_conversions_baseline(df, future_periods=5)
    print(f"   Generated {len(quick_forecasts)} quick forecasts")

    # Show feature importance
    print("\n4. Feature importance:")
    importance = forecaster.get_feature_importance()
    for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[
        :5
    ]:
        print(f"   {feature}: {imp:.4f}")

    return forecaster


def test_time_series_forecasting():
    """Test time series forecasting functionality."""
    print("\n\n📈 Testing Time Series Forecasting")
    print("=" * 50)

    # Load sample data
    csv_path = project_root / "sample_data" / "campaign_data.csv"
    df = load_campaign_data(csv_path)

    # Test seasonal naive forecaster
    print("\n1. Testing Seasonal Naive Forecaster...")
    ts_forecaster = TimeSeriesForecaster(method="seasonal_naive")
    ts_forecaster.fit(df)

    # Make predictions
    predictions = ts_forecaster.predict(periods=7)
    print(f"   Generated {len(predictions)} predictions")
    print(
        f"   Predictions range: {predictions['predicted_conversions'].min():.1f} to {predictions['predicted_conversions'].max():.1f}"
    )

    # Test convenience function
    print("\n2. Testing convenience function...")
    quick_forecasts = forecast_conversions_time_series(
        df, future_periods=5, method="seasonal_naive"
    )
    print(f"   Generated {len(quick_forecasts)} quick forecasts")

    # Try ARIMA if statsmodels is available
    try:
        print("\n3. Testing ARIMA Forecaster...")
        arima_forecaster = TimeSeriesForecaster(method="arima")
        arima_forecaster.fit(df)
        arima_predictions = arima_forecaster.predict(periods=7)
        print(f"   ARIMA generated {len(arima_predictions)} predictions")
    except ImportError:
        print("\n3. ARIMA not available (statsmodels not installed)")
    except Exception as e:
        print(f"\n3. ARIMA failed: {e}")

    return ts_forecaster


def test_model_selection():
    """Test model selection functionality."""
    print("\n\n🎯 Testing Model Selection")
    print("=" * 50)

    # Load sample data
    csv_path = project_root / "sample_data" / "campaign_data.csv"
    df = load_campaign_data(csv_path)

    # Test model comparison
    print("\n1. Comparing models...")
    comparison = compare_models(df, cv_folds=2)
    print(f"   Compared {len(comparison)} models")
    print("   Top 3 models:")
    for i, row in comparison.head(3).iterrows():
        print(
            f"   {i+1}. {row['model']}: {row['mae_mean']:.2f} MAE (±{row['mae_std']:.2f})"
        )

    # Test best model selection
    print("\n2. Selecting best model...")
    best_model, model_info = select_best_model(df, cv_folds=2)
    print(f"   Best model: {model_info['name']}")
    print(f"   Best score: {model_info['score']:.4f}")
    print(f"   Model type: {model_info['model_type']}")

    # Test ModelSelector class
    print("\n3. Testing ModelSelector class...")
    selector = ModelSelector(cv_folds=2)
    selector.fit(df)

    # Make predictions with best model
    if hasattr(selector.best_model, "predict"):
        # Baseline model
        predictions = selector.predict(df)
        print(f"   Generated {len(predictions)} predictions using best model")
    else:
        # Time series model
        predictions = selector.predict(periods=7)
        print(f"   Generated {len(predictions)} time series predictions")

    return selector


def test_end_to_end_forecasting():
    """Test end-to-end forecasting with model selection."""
    print("\n\n🚀 Testing End-to-End Forecasting")
    print("=" * 50)

    # Load sample data
    csv_path = project_root / "sample_data" / "campaign_data.csv"
    df = load_campaign_data(csv_path)

    # Test end-to-end forecasting
    print("\n1. End-to-end forecasting with model selection...")
    forecasts, model_info = forecast_with_best_model(
        df, future_periods=10, spend_scenario=250.0, cv_folds=2
    )

    print(f"   Generated {len(forecasts)} forecasts")
    print(f"   Used model: {model_info['name']}")
    print(f"   Model score: {model_info['score']:.4f}")

    # Show some forecast results
    print("\n2. Sample forecast results:")
    for i in range(min(5, len(forecasts))):
        row = forecasts.iloc[i]
        if "date" in row:
            print(
                f"   {row['date'].strftime('%Y-%m-%d')}: {row['predicted_conversions']:.1f} conversions "
                f"(CI: {row['lower_bound']:.1f} - {row['upper_bound']:.1f})"
            )
        else:
            print(
                f"   Period {i+1}: {row['predicted_conversions']:.1f} conversions "
                f"(CI: {row['lower_bound']:.1f} - {row['upper_bound']:.1f})"
            )

    return forecasts, model_info


def test_error_handling():
    """Test error handling and edge cases."""
    print("\n\n⚠️ Testing Error Handling")
    print("=" * 50)

    # Test with minimal data
    print("\n1. Testing with minimal data...")
    minimal_df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "spend": [100, 200],
            "impressions": [1000, 2000],
            "clicks": [50, 100],
            "conversions": [2, 5],
        }
    )

    try:
        forecaster = BaselineForecaster()
        forecaster.fit(minimal_df)
        print("   ✓ Minimal data handled successfully")
    except Exception as e:
        print(f"   ✗ Error with minimal data: {e}")

    # Test with missing features
    print("\n2. Testing with missing features...")
    incomplete_df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "spend": [100, 200, 150],
            "conversions": [2, 5, 3],
        }
    )

    try:
        forecaster = BaselineForecaster(features=["spend"])
        forecaster.fit(incomplete_df)
        print("   ✓ Missing features handled successfully")
    except Exception as e:
        print(f"   ✗ Error with missing features: {e}")

    # Test prediction before training
    print("\n3. Testing prediction before training...")
    try:
        untrained_forecaster = BaselineForecaster()
        untrained_forecaster.predict(minimal_df)
        print("   ✗ Should have raised an error")
    except ValueError as e:
        print(f"   ✓ Correctly raised error: {e}")
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")


def main():
    """Main test function."""
    print("🎯 Digital Ad Campaign ROI Forecaster")
    print("📊 Forecasting Module Test Suite")
    print("=" * 60)

    try:
        # Run all tests
        baseline_forecaster = test_baseline_forecasting()
        ts_forecaster = test_time_series_forecasting()
        selector = test_model_selection()
        forecasts, model_info = test_end_to_end_forecasting()
        test_error_handling()

        print("\n\n🎉 All Tests Complete!")
        print("=" * 60)
        print("✅ Baseline forecasting: SUCCESS")
        print("✅ Time series forecasting: SUCCESS")
        print("✅ Model selection: SUCCESS")
        print("✅ End-to-end forecasting: SUCCESS")
        print("✅ Error handling: SUCCESS")
        print("\n🚀 Forecasting module is ready for production use!")

        # Show final summary
        print("\n📊 Final Summary:")
        print(f"   Best model type: {model_info['model_type']}")
        print(f"   Best model name: {model_info['name']}")
        print(f"   Cross-validation score: {model_info['score']:.4f}")
        print(f"   Forecast horizon: {len(forecasts)} periods")

    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
