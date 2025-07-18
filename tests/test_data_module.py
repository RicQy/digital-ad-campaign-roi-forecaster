# test_data_module.py
import pytest
import pandas as pd
from ad_roi_forecaster.data.loader import CampaignDataLoader, load_campaign_data
from ad_roi_forecaster.data.schemas import CampaignRecord, CampaignDataset
from ad_roi_forecaster.data.validator import validate_campaign_data

# Sample data for testing
def test_load_and_validate_campaign_data():
    # Let's read and validate sample data
    sample_data_path = "./sample_data/sample_campaigns.csv"
    
    # Load campaign data
    campaign_data = load_campaign_data(sample_data_path)
    
    # Validate data
    validation_result = validate_campaign_data(campaign_data)
    
    assert validation_result.is_valid, validation_result.errors
    assert len(campaign_data) > 0, "No data loaded"

    # Test accessing data structures
    dataset = CampaignDataset(records=[CampaignRecord(**record) for record in campaign_data.to_dict(orient='records')])
    assert len(dataset.records) == len(campaign_data)
