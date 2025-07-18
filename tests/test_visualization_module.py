# test_visualization_module.py
import pytest
import pandas as pd
import matplotlib.pyplot as plt
from ad_roi_forecaster.visualization.plots import (
    plot_spend_vs_conversions,
    plot_forecasted_conversions,
)
from ad_roi_forecaster.visualization.report import ReportGenerator


# Sample data for testing
def sample_campaign_df():
    data = {
        "spend": [100, 200, 300, 400, 500],
        "conversions": [4, 8, 12, 16, 20],
        "revenue": [200, 400, 600, 800, 1000],
        "platform": ["google", "facebook", "instagram", "twitter", "linkedin"],
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        "clicks": [50, 100, 150, 200, 250],
        "impressions": [1000, 2000, 3000, 4000, 5000],
    }
    return pd.DataFrame(data)


def sample_forecast_df():
    data = {
        "date": ["2024-01-06", "2024-01-07", "2024-01-08"],
        "conversions": [22, 24, 26],
    }
    return pd.DataFrame(data)


def test_plot_spend_vs_conversions():
    """Test spend vs conversions plot creation."""
    df = sample_campaign_df()

    # Test plot creation without showing/saving
    fig = plot_spend_vs_conversions(df, show_plot=False, save_path=None)

    assert fig is not None, "Figure should be created"
    assert len(fig.axes) == 1, "Should have one axis"

    # Clean up
    plt.close(fig)


def test_plot_forecasted_conversions():
    """Test forecasted conversions plot creation."""
    historical_df = sample_campaign_df()
    forecast_df = sample_forecast_df()

    # Test plot creation without showing/saving
    fig = plot_forecasted_conversions(
        historical_df, forecast_df, show_plot=False, save_path=None
    )

    assert fig is not None, "Figure should be created"
    assert len(fig.axes) == 1, "Should have one axis"

    # Clean up
    plt.close(fig)


def test_report_generator():
    """Test report generator initialization and basic functionality."""
    df = sample_campaign_df()

    # Test report generator initialization
    report_gen = ReportGenerator()
    assert report_gen.config is not None, "Config should be initialized"

    # Test summary stats calculation
    report_gen._calculate_summary_stats(df)
    assert report_gen.summary_stats is not None, "Summary stats should be calculated"
    assert report_gen.summary_stats["total_campaigns"] == len(
        df
    ), "Total campaigns should match"
    assert (
        report_gen.summary_stats["total_spend"] == df["spend"].sum()
    ), "Total spend should match"
