# test_optimization_module.py
import pytest
from ad_roi_forecaster.optimization.roi_optimizer import optimize_roi
from ad_roi_forecaster.data.schemas import CampaignRecord, CampaignDataset
from datetime import datetime


def sample_campaign_dataset():
    """Create a sample campaign dataset for testing."""
    records = [
        CampaignRecord(
            campaign_id="CAMP001",
            campaign_name="Test Campaign 1",
            campaign_type="search",
            platform="google",
            date=datetime(2024, 1, 1),
            spend=100.0,
            impressions=1000,
            clicks=50,
            conversions=5,
            revenue=250.0,
            target_audience="Young Adults",
            geographic_region="US",
            budget_daily=150.0,
            budget_total=3000.0
        ),
        CampaignRecord(
            campaign_id="CAMP002",
            campaign_name="Test Campaign 2",
            campaign_type="display",
            platform="facebook",
            date=datetime(2024, 1, 2),
            spend=200.0,
            impressions=2000,
            clicks=100,
            conversions=10,
            revenue=500.0,
            target_audience="Millennials",
            geographic_region="US",
            budget_daily=250.0,
            budget_total=5000.0
        ),
        CampaignRecord(
            campaign_id="CAMP003",
            campaign_name="Test Campaign 3",
            campaign_type="video",
            platform="youtube",
            date=datetime(2024, 1, 3),
            spend=300.0,
            impressions=3000,
            clicks=150,
            conversions=15,
            revenue=750.0,
            target_audience="Gen Z",
            geographic_region="CA",
            budget_daily=350.0,
            budget_total=7000.0
        )
    ]
    return CampaignDataset(records=records)


def test_optimize_roi():
    """Test the ROI optimization function."""
    dataset = sample_campaign_dataset()
    total_budget = 1000.0
    
    # Run optimization
    allocation, expected_roi = optimize_roi(dataset, total_budget)
    
    # Check that allocation is returned
    assert allocation is not None, "Allocation should not be None"
    assert len(allocation) == len(dataset.records), "Allocation should have same length as records"
    
    # Check that expected ROI is calculated
    assert expected_roi is not None, "Expected ROI should not be None"
    assert isinstance(expected_roi, (int, float)), "Expected ROI should be numeric"
    
    # Check that allocation sums to total budget (within tolerance)
    assert abs(sum(allocation) - total_budget) < 1e-6, "Allocation should sum to total budget"
    
    # Check that all allocations are non-negative
    assert all(alloc >= 0 for alloc in allocation), "All allocations should be non-negative"


def test_optimize_roi_with_zero_budget():
    """Test ROI optimization with zero budget."""
    dataset = sample_campaign_dataset()
    total_budget = 0.0
    
    # Run optimization
    allocation, expected_roi = optimize_roi(dataset, total_budget)
    
    # Check that allocation is all zeros
    assert all(alloc == 0 for alloc in allocation), "All allocations should be zero with zero budget"
    
    # Check that allocation sums to zero
    assert sum(allocation) == 0, "Allocation should sum to zero budget"


def test_optimize_roi_with_large_budget():
    """Test ROI optimization with large budget."""
    dataset = sample_campaign_dataset()
    total_budget = 100000.0
    
    # Run optimization
    allocation, expected_roi = optimize_roi(dataset, total_budget)
    
    # Check that allocation is returned
    assert allocation is not None, "Allocation should not be None"
    assert len(allocation) == len(dataset.records), "Allocation should have same length as records"
    
    # Check that allocation sums to total budget (within tolerance)
    assert abs(sum(allocation) - total_budget) < 1e-6, "Allocation should sum to total budget"
