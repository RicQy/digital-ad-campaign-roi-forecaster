# test_forecasting_module.py
import pytest
import pandas as pd
from ad_roi_forecaster.forecasting.baseline import BaselineForecaster
from ad_roi_forecaster.forecasting.model_selector import ModelSelector

# Sample data for testing
def sample_campaign_df():
    data = {
        "spend": [100, 200, 300],
        "impressions": [1000, 1500, 2000],
        "clicks": [100, 120, 150],
        "conversions": [4, 5, 6],
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
    }
    return pd.DataFrame(data)

def test_baseline_forecaster():
    df = sample_campaign_df()
    model = BaselineForecaster(features=["spend", "impressions", "clicks"])
    model.fit(df)
    predictions = model.predict(df)
    assert not predictions.empty, "Predictions should not be empty"

    # Check if model attributes are populated
    assert model.is_trained, "Model should be trained"
    assert model.train_score is not None, "Training score should be calculated"

def test_model_selector():
    df = sample_campaign_df()
    model_selector = ModelSelector()
    model_selector.fit(df)

    # Ensure the best model is selected
    assert model_selector.best_model is not None, "There should be a best model selected"
    assert model_selector.best_model_name is not None, "Best model name should be available"
