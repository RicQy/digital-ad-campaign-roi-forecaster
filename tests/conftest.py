# conftest.py
import pytest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configuration for pytest
pytest_plugins = []

# Common fixtures can be defined here
@pytest.fixture
def sample_data():
    """Fixture providing sample campaign data."""
    import pandas as pd
    
    data = {
        "campaign_id": ["CAMP001", "CAMP002", "CAMP003"],
        "campaign_name": ["Test Campaign 1", "Test Campaign 2", "Test Campaign 3"],
        "campaign_type": ["search", "display", "video"],
        "platform": ["google", "facebook", "youtube"],
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "spend": [100, 200, 300],
        "impressions": [1000, 2000, 3000],
        "clicks": [50, 100, 150],
        "conversions": [5, 10, 15],
        "revenue": [250, 500, 750],
        "target_audience": ["Young Adults", "Millennials", "Gen Z"],
        "geographic_region": ["US", "US", "CA"],
        "budget_daily": [150, 250, 350],
        "budget_total": [3000, 5000, 7000]
    }
    return pd.DataFrame(data)
