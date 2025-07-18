"""
Pydantic models for campaign data validation and structure.

This module defines the data schemas used for validating and structuring
digital advertising campaign data.
"""

from datetime import datetime
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum


class CampaignType(str, Enum):
    """Campaign type enumeration."""

    SEARCH = "search"
    DISPLAY = "display"
    VIDEO = "video"
    SOCIAL = "social"
    SHOPPING = "shopping"


class Platform(str, Enum):
    """Advertising platform enumeration."""

    GOOGLE = "google"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"


class CampaignRecord(BaseModel):
    """
    Schema for a single campaign record.

    Represents one row of campaign data with all required fields
    for ROI forecasting.
    """

    # Campaign identifiers
    campaign_id: str = Field(..., description="Unique campaign identifier")
    campaign_name: str = Field(..., description="Human-readable campaign name")

    # Campaign metadata
    campaign_type: CampaignType = Field(..., description="Type of campaign")
    platform: Platform = Field(..., description="Advertising platform")

    # Time period
    date: datetime = Field(..., description="Campaign date")

    # Spend and performance metrics
    spend: float = Field(..., ge=0, description="Campaign spend amount")
    impressions: int = Field(..., ge=0, description="Number of impressions")
    clicks: int = Field(..., ge=0, description="Number of clicks")
    conversions: int = Field(..., ge=0, description="Number of conversions")

    # Revenue metrics
    revenue: float = Field(..., ge=0, description="Revenue generated")

    # Optional targeting information
    target_audience: Optional[str] = Field(None, description="Target audience segment")
    geographic_region: Optional[str] = Field(
        None, description="Geographic targeting region"
    )

    # Optional campaign settings
    budget_daily: Optional[float] = Field(None, ge=0, description="Daily budget limit")
    budget_total: Optional[float] = Field(
        None, ge=0, description="Total campaign budget"
    )

    @model_validator(mode="after")
    def validate_campaign_metrics(self):
        """Validate campaign metrics consistency."""
        # Validate that clicks don't exceed impressions
        if self.clicks > self.impressions:
            raise ValueError("Clicks cannot exceed impressions")

        # Validate that conversions don't exceed clicks
        if self.conversions > self.clicks:
            raise ValueError("Conversions cannot exceed clicks")

        # Validate that spend is reasonable relative to revenue
        if self.spend > self.revenue * 10:
            raise ValueError("Spend seems unreasonably high compared to revenue")

        # Validate budget constraints
        if self.budget_daily is not None:
            if self.spend > self.budget_daily * 1.1:  # Allow 10% overspend tolerance
                raise ValueError("Daily spend significantly exceeds daily budget")

        if self.budget_total is not None:
            if self.spend > self.budget_total:
                raise ValueError("Spend exceeds total budget")

        return self

    @property
    def ctr(self) -> float:
        """Calculate click-through rate."""
        if self.impressions == 0:
            return 0.0
        return self.clicks / self.impressions

    @property
    def conversion_rate(self) -> float:
        """Calculate conversion rate."""
        if self.clicks == 0:
            return 0.0
        return self.conversions / self.clicks

    @property
    def cpc(self) -> float:
        """Calculate cost per click."""
        if self.clicks == 0:
            return 0.0
        return self.spend / self.clicks

    @property
    def cpa(self) -> float:
        """Calculate cost per acquisition."""
        if self.conversions == 0:
            return 0.0
        return self.spend / self.conversions

    @property
    def roi(self) -> float:
        """Calculate return on investment."""
        if self.spend == 0:
            return 0.0
        return (self.revenue - self.spend) / self.spend

    @property
    def roas(self) -> float:
        """Calculate return on ad spend."""
        if self.spend == 0:
            return 0.0
        return self.revenue / self.spend


class CampaignDataset(BaseModel):
    """
    Schema for a complete campaign dataset.

    Contains multiple campaign records with dataset-level validation.
    """

    records: List[CampaignRecord] = Field(..., description="List of campaign records")

    @field_validator("records")
    @classmethod
    def records_not_empty(cls, v):
        """Validate that records list is not empty."""
        if not v:
            raise ValueError("Dataset must contain at least one record")
        return v

    @model_validator(mode="after")
    def validate_dataset_consistency(self):
        """Validate dataset-level consistency."""
        if not self.records:
            return self

        # Check for duplicate campaign_id + date combinations
        seen_combinations = set()
        for record in self.records:
            combination = (record.campaign_id, record.date.date())
            if combination in seen_combinations:
                raise ValueError(
                    f"Duplicate campaign_id and date combination: {combination}"
                )
            seen_combinations.add(combination)

        return self

    @property
    def total_spend(self) -> float:
        """Calculate total spend across all records."""
        return sum(record.spend for record in self.records)

    @property
    def total_revenue(self) -> float:
        """Calculate total revenue across all records."""
        return sum(record.revenue for record in self.records)

    @property
    def overall_roi(self) -> float:
        """Calculate overall ROI for the entire dataset."""
        if self.total_spend == 0:
            return 0.0
        return (self.total_revenue - self.total_spend) / self.total_spend

    @property
    def date_range(self) -> tuple[datetime, datetime]:
        """Get the date range of the dataset."""
        if not self.records:
            raise ValueError("No records in dataset")

        dates = [record.date for record in self.records]
        return min(dates), max(dates)


class ValidationResult(BaseModel):
    """
    Schema for validation results.

    Contains information about data validation outcomes.
    """

    is_valid: bool = Field(..., description="Whether the data is valid")
    errors: List[str] = Field(
        default_factory=list, description="List of validation errors"
    )
    warnings: List[str] = Field(
        default_factory=list, description="List of validation warnings"
    )
    record_count: int = Field(..., description="Number of records processed")

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0
