"""
Data Input & Validation module for campaign data processing.

This module provides functionality to load, validate, and process
digital advertising campaign data from CSV files.

Main functions:
- load_campaign_data: Load campaign data from CSV files
- validate_campaign_data: Validate campaign data quality

Classes:
- CampaignDataLoader: CSV loading and processing
- CampaignDataValidator: Data validation and quality checks
- CampaignRecord: Schema for individual campaign records
- CampaignDataset: Schema for complete campaign datasets
"""

from .loader import load_campaign_data, CampaignDataLoader
from .validator import validate_campaign_data, CampaignDataValidator, DataValidator
from .schemas import (
    CampaignRecord,
    CampaignDataset,
    ValidationResult,
    CampaignType,
    Platform,
)

__all__ = [
    # Main functions
    "load_campaign_data",
    "validate_campaign_data",
    # Classes
    "CampaignDataLoader",
    "CampaignDataValidator",
    "DataValidator",
    "CampaignRecord",
    "CampaignDataset",
    "ValidationResult",
    "CampaignType",
    "Platform",
]
