#!/usr/bin/env python3
"""
Test script for the data input and validation module.

This script demonstrates the functionality of the data module by:
1. Loading campaign data from CSV
2. Validating the data
3. Displaying results
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ad_roi_forecaster.data import load_campaign_data, validate_campaign_data


def test_data_loading():
    """Test the data loading functionality."""
    print("=" * 60)
    print("Testing Data Loading Functionality")
    print("=" * 60)

    try:
        # Load the sample data
        csv_path = project_root / "sample_data" / "campaign_data.csv"

        print(f"Loading data from: {csv_path}")
        df = load_campaign_data(csv_path)

        print(f"✓ Successfully loaded {len(df)} records")
        print(f"✓ Columns: {list(df.columns)}")
        print(f"✓ Data types:")
        for col, dtype in df.dtypes.items():
            print(f"  - {col}: {dtype}")

        print("\nFirst 3 rows:")
        print(df.head(3))

        return df

    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None


def test_data_validation(df):
    """Test the data validation functionality."""
    print("\n" + "=" * 60)
    print("Testing Data Validation Functionality")
    print("=" * 60)

    try:
        # Validate the data
        validation_result = validate_campaign_data(df)

        print(f"✓ Validation completed")
        print(f"✓ Is valid: {validation_result.is_valid}")
        print(f"✓ Records processed: {validation_result.record_count}")

        if validation_result.errors:
            print(f"✗ Errors found ({len(validation_result.errors)}):")
            for error in validation_result.errors:
                print(f"  - {error}")
        else:
            print("✓ No errors found")

        if validation_result.warnings:
            print(f"⚠ Warnings found ({len(validation_result.warnings)}):")
            for warning in validation_result.warnings:
                print(f"  - {warning}")
        else:
            print("✓ No warnings found")

        return validation_result

    except Exception as e:
        print(f"✗ Error validating data: {e}")
        return None


def test_calculated_metrics(df):
    """Test calculated metrics."""
    print("\n" + "=" * 60)
    print("Testing Calculated Metrics")
    print("=" * 60)

    try:
        # Calculate some basic metrics
        if all(
            col in df.columns
            for col in ["spend", "revenue", "clicks", "impressions", "conversions"]
        ):
            total_spend = df["spend"].sum()
            total_revenue = df["revenue"].sum()
            total_clicks = df["clicks"].sum()
            total_impressions = df["impressions"].sum()
            total_conversions = df["conversions"].sum()

            # Calculate derived metrics
            overall_roi = (
                (total_revenue - total_spend) / total_spend if total_spend > 0 else 0
            )
            overall_roas = total_revenue / total_spend if total_spend > 0 else 0
            overall_ctr = (
                total_clicks / total_impressions if total_impressions > 0 else 0
            )
            overall_cvr = total_conversions / total_clicks if total_clicks > 0 else 0

            print(f"✓ Total Spend: ${total_spend:,.2f}")
            print(f"✓ Total Revenue: ${total_revenue:,.2f}")
            print(f"✓ Total Clicks: {total_clicks:,}")
            print(f"✓ Total Impressions: {total_impressions:,}")
            print(f"✓ Total Conversions: {total_conversions:,}")
            print(f"✓ Overall ROI: {overall_roi:.2%}")
            print(f"✓ Overall ROAS: {overall_roas:.2f}")
            print(f"✓ Overall CTR: {overall_ctr:.2%}")
            print(f"✓ Overall CVR: {overall_cvr:.2%}")

    except Exception as e:
        print(f"✗ Error calculating metrics: {e}")


def main():
    """Main test function."""
    print("Digital Ad Campaign ROI Forecaster - Data Module Test")
    print("=" * 60)

    # Test data loading
    df = test_data_loading()

    if df is not None:
        # Test data validation
        validation_result = test_data_validation(df)

        # Test calculated metrics
        test_calculated_metrics(df)

        print("\n" + "=" * 60)
        print("✓ All tests completed successfully!")
        print("✓ Data module is ready for use")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗ Tests failed - data could not be loaded")
        print("=" * 60)


if __name__ == "__main__":
    main()
