"""
Validator module for campaign data quality and consistency checks.

This module provides functionality to validate campaign data for missing
columns, date ranges, numeric bounds, and data coherence.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from .schemas import CampaignDataset, ValidationResult, Platform, CampaignType


class CampaignDataValidator:
    """
    Comprehensive validator for campaign data quality and consistency.
    
    Performs validation checks for missing columns, date ranges, numeric bounds,
    data coherence, and business logic consistency.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the validator with a DataFrame.
        
        Args:
            data: DataFrame with campaign data to validate.
        """
        self.data = data
        self.errors = []
        self.warnings = []

    def _check_required_columns(self) -> List[str]:
        """
        Check for presence of all required columns.
        
        Returns:
            List of missing columns, if any
        """
        required_columns = ['campaign_id', 'campaign_name', 'campaign_type', 'platform',
                            'date', 'spend', 'impressions', 'clicks', 'conversions', 'revenue']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        return missing_columns

    def _check_date_range(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> List[str]:
        """
        Check that all dates fall within an optional specified range.
        
        Args:
            start_date: Optional minimum date
            end_date: Optional maximum date
            
        Returns:
            List of out-of-bound date errors
        """
        errors = []
        
        if 'date' not in self.data.columns:
            return errors
        
        # Check for null dates
        null_dates = self.data['date'].isna().sum()
        if null_dates > 0:
            errors.append(f"{null_dates} records with missing dates")
        
        # Check for invalid dates
        try:
            valid_dates = pd.to_datetime(self.data['date'], errors='coerce')
            invalid_dates = valid_dates.isna().sum() - null_dates
            if invalid_dates > 0:
                errors.append(f"{invalid_dates} records with invalid date formats")
        except Exception:
            errors.append("Unable to parse dates in the date column")
        
        if start_date:
            out_of_bounds_start = self.data[self.data['date'] < start_date]
            if not out_of_bounds_start.empty:
                errors.append(f"{len(out_of_bounds_start)} records with dates before {start_date}")
        
        if end_date:
            out_of_bounds_end = self.data[self.data['date'] > end_date]
            if not out_of_bounds_end.empty:
                errors.append(f"{len(out_of_bounds_end)} records with dates after {end_date}")
        
        return errors
    
    def _check_numeric_bounds(self) -> List[str]:
        """
        Check numeric fields for unreasonable values and consistency.
        
        Returns:
            List of numeric validation errors and warnings
        """
        issues = []
        
        # Check for negative values
        numeric_columns = ['spend', 'impressions', 'clicks', 'conversions', 'revenue']
        
        for col in numeric_columns:
            if col in self.data.columns:
                negative_count = (self.data[col] < 0).sum()
                if negative_count > 0:
                    issues.append(f"{col} contains {negative_count} negative values")
        
        # Check logical consistency
        if all(col in self.data.columns for col in ['clicks', 'impressions']):
            invalid_ctr = (self.data['clicks'] > self.data['impressions']).sum()
            if invalid_ctr > 0:
                issues.append(f"{invalid_ctr} records where clicks exceed impressions")
        
        if all(col in self.data.columns for col in ['conversions', 'clicks']):
            invalid_cvr = (self.data['conversions'] > self.data['clicks']).sum()
            if invalid_cvr > 0:
                issues.append(f"{invalid_cvr} records where conversions exceed clicks")
        
        # Check for extreme values (warnings)
        if 'spend' in self.data.columns and 'revenue' in self.data.columns:
            # Check for extremely high spend vs revenue ratio
            high_spend_ratio = (self.data['spend'] > self.data['revenue'] * 10).sum()
            if high_spend_ratio > 0:
                issues.append(f"{high_spend_ratio} records with spend more than 10x revenue")
        
        # Check for zero values that might indicate data issues
        if 'impressions' in self.data.columns:
            zero_impressions = (self.data['impressions'] == 0).sum()
            if zero_impressions > 0:
                issues.append(f"{zero_impressions} records with zero impressions")
        
        return issues
    
    def _check_data_completeness(self) -> List[str]:
        """
        Check for missing values in important columns.
        
        Returns:
            List of data completeness warnings
        """
        warnings = []
        
        # Check for missing values in key columns
        key_columns = ['campaign_id', 'campaign_name', 'platform', 'campaign_type']
        
        for col in key_columns:
            if col in self.data.columns:
                missing_count = self.data[col].isna().sum()
                if missing_count > 0:
                    warnings.append(f"{missing_count} records missing {col}")
        
        return warnings
    
    def _check_data_consistency(self) -> List[str]:
        """
        Check for data consistency issues.
        
        Returns:
            List of data consistency warnings
        """
        warnings = []
        
        # Check for duplicate campaign_id + date combinations
        if all(col in self.data.columns for col in ['campaign_id', 'date']):
            duplicates = self.data.duplicated(subset=['campaign_id', 'date']).sum()
            if duplicates > 0:
                warnings.append(f"{duplicates} duplicate campaign_id + date combinations")
        
        # Check for valid platform values
        if 'platform' in self.data.columns:
            valid_platforms = {p.value for p in Platform}
            invalid_platforms = ~self.data['platform'].str.lower().isin(valid_platforms)
            invalid_count = invalid_platforms.sum()
            if invalid_count > 0:
                warnings.append(f"{invalid_count} records with invalid platform values")
        
        # Check for valid campaign type values
        if 'campaign_type' in self.data.columns:
            valid_types = {t.value for t in CampaignType}
            invalid_types = ~self.data['campaign_type'].str.lower().isin(valid_types)
            invalid_count = invalid_types.sum()
            if invalid_count > 0:
                warnings.append(f"{invalid_count} records with invalid campaign type values")
        
        return warnings
    
    def _check_statistical_outliers(self) -> List[str]:
        """
        Check for statistical outliers in numeric columns.
        
        Returns:
            List of statistical outlier warnings
        """
        warnings = []
        
        numeric_columns = ['spend', 'impressions', 'clicks', 'conversions', 'revenue']
        
        for col in numeric_columns:
            if col in self.data.columns:
                # Use IQR method to detect outliers
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((self.data[col] < lower_bound) | (self.data[col] > upper_bound)).sum()
                
                if outliers > 0:
                    warnings.append(f"{outliers} statistical outliers detected in {col}")
        
        return warnings
    
    def validate(self, 
                 start_date: Optional[datetime] = None, 
                 end_date: Optional[datetime] = None,
                 check_outliers: bool = True) -> ValidationResult:
        """
        Perform comprehensive validation of the data.
        
        Args:
            start_date: Optional minimum date for date range check
            end_date: Optional maximum date for date range check
            check_outliers: Whether to check for statistical outliers
            
        Returns:
            ValidationResult object with validation outcome
        """
        errors = []
        warnings = []
        
        # Check for empty dataset
        if self.data.empty:
            errors.append("Dataset is empty")
            return ValidationResult(
                is_valid=False, 
                errors=errors, 
                warnings=warnings, 
                record_count=0
            )
        
        # Check for required columns
        missing_columns = self._check_required_columns()
        if missing_columns:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Check date ranges
        date_errors = self._check_date_range(start_date, end_date)
        errors.extend(date_errors)
        
        # Check numeric bounds and consistency
        numeric_issues = self._check_numeric_bounds()
        # Separate errors from warnings based on severity
        for issue in numeric_issues:
            if any(keyword in issue.lower() for keyword in ['negative', 'exceed', 'invalid']):
                errors.append(issue)
            else:
                warnings.append(issue)
        
        # Check data completeness
        completeness_warnings = self._check_data_completeness()
        warnings.extend(completeness_warnings)
        
        # Check data consistency
        consistency_warnings = self._check_data_consistency()
        warnings.extend(consistency_warnings)
        
        # Check for statistical outliers if requested
        if check_outliers:
            outlier_warnings = self._check_statistical_outliers()
            warnings.extend(outlier_warnings)
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            record_count=len(self.data)
        )


# Maintain backward compatibility
DataValidator = CampaignDataValidator


def validate_campaign_data(data: pd.DataFrame, **kwargs) -> ValidationResult:
    """
    Convenience function to validate campaign data.
    
    Args:
        data: DataFrame with campaign data to validate
        **kwargs: Additional arguments for validation
        
    Returns:
        ValidationResult object with validation outcome
    """
    validator = CampaignDataValidator(data)
    return validator.validate(**kwargs)
