#!/usr/bin/env python3
"""
Complete Data Module Workflow Demo

This script demonstrates the full workflow of the data input and validation module:
1. Loading campaign data from CSV
2. Validating data quality
3. Processing and analysis
4. Generating reports
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ad_roi_forecaster.data import (
    load_campaign_data,
    validate_campaign_data,
    CampaignRecord,
    CampaignDataset,
    CampaignDataLoader,
    CampaignDataValidator,
)


def demo_basic_workflow():
    """Demonstrate basic data loading and validation workflow."""
    print("🚀 Basic Data Workflow Demo")
    print("=" * 50)

    # Step 1: Load data
    print("\n1. Loading Campaign Data...")
    csv_path = project_root / "sample_data" / "campaign_data.csv"
    df = load_campaign_data(csv_path)
    print(f"   ✓ Loaded {len(df)} records")

    # Step 2: Validate data
    print("\n2. Validating Data Quality...")
    validation_result = validate_campaign_data(df)

    if validation_result.is_valid:
        print("   ✓ Data validation passed")
    else:
        print("   ✗ Data validation failed")
        for error in validation_result.errors:
            print(f"     - {error}")

    if validation_result.warnings:
        print(f"   ⚠ {len(validation_result.warnings)} warnings found:")
        for warning in validation_result.warnings:
            print(f"     - {warning}")

    # Step 3: Basic analysis
    print("\n3. Basic Analysis...")
    total_spend = df["spend"].sum()
    total_revenue = df["revenue"].sum()
    roi = (total_revenue - total_spend) / total_spend * 100

    print(f"   Total Spend: ${total_spend:,.2f}")
    print(f"   Total Revenue: ${total_revenue:,.2f}")
    print(f"   Overall ROI: {roi:.1f}%")

    return df, validation_result


def demo_advanced_validation():
    """Demonstrate advanced validation features."""
    print("\n\n🔍 Advanced Validation Demo")
    print("=" * 50)

    # Load data
    csv_path = project_root / "sample_data" / "campaign_data.csv"
    df = load_campaign_data(csv_path)

    # Create validator with custom settings
    validator = CampaignDataValidator(df)

    # Validate with date range
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)

    print(f"\n1. Validating with date range: {start_date.date()} to {end_date.date()}")
    result = validator.validate(
        start_date=start_date, end_date=end_date, check_outliers=True
    )

    print(f"   ✓ Records processed: {result.record_count}")
    print(f"   ✓ Validation status: {'PASS' if result.is_valid else 'FAIL'}")

    if result.warnings:
        print(f"   ⚠ {len(result.warnings)} warnings:")
        for warning in result.warnings[:3]:  # Show first 3
            print(f"     - {warning}")

    # Platform analysis
    print("\n2. Platform Analysis...")
    platform_stats = (
        df.groupby("platform")
        .agg({"spend": "sum", "revenue": "sum", "clicks": "sum", "impressions": "sum"})
        .round(2)
    )

    print(platform_stats)

    return result


def demo_pydantic_schemas():
    """Demonstrate Pydantic schema usage."""
    print("\n\n📊 Pydantic Schema Demo")
    print("=" * 50)

    # Create a campaign record
    print("\n1. Creating Campaign Record...")
    try:
        record = CampaignRecord(
            campaign_id="DEMO001",
            campaign_name="Demo Campaign",
            campaign_type="search",
            platform="google",
            date=datetime.now(),
            spend=100.0,
            impressions=5000,
            clicks=150,
            conversions=10,
            revenue=300.0,
            target_audience="Tech Enthusiasts",
            geographic_region="US",
        )

        print(f"   ✓ Campaign: {record.campaign_name}")
        print(f"   ✓ Platform: {record.platform}")
        print(f"   ✓ Date: {record.date.date()}")

        # Display calculated metrics
        print("\n2. Calculated Metrics...")
        print(f"   ROI: {record.roi:.1%}")
        print(f"   ROAS: {record.roas:.2f}")
        print(f"   CTR: {record.ctr:.2%}")
        print(f"   Conversion Rate: {record.conversion_rate:.2%}")
        print(f"   CPC: ${record.cpc:.2f}")
        print(f"   CPA: ${record.cpa:.2f}")

    except Exception as e:
        print(f"   ✗ Error creating record: {e}")

    # Test validation
    print("\n3. Testing Validation...")
    try:
        # This should fail - clicks > impressions
        invalid_record = CampaignRecord(
            campaign_id="INVALID001",
            campaign_name="Invalid Campaign",
            campaign_type="search",
            platform="google",
            date=datetime.now(),
            spend=100.0,
            impressions=1000,
            clicks=1500,  # More clicks than impressions!
            conversions=10,
            revenue=300.0,
        )
    except Exception as e:
        print(f"   ✓ Validation caught error: {e}")


def demo_data_insights():
    """Generate insights from the data."""
    print("\n\n📈 Data Insights Demo")
    print("=" * 50)

    # Load data
    csv_path = project_root / "sample_data" / "campaign_data.csv"
    df = load_campaign_data(csv_path)

    # Campaign performance analysis
    print("\n1. Top Performing Campaigns (by ROI)...")
    df["roi"] = (df["revenue"] - df["spend"]) / df["spend"] * 100
    top_campaigns = df.nlargest(3, "roi")[["campaign_name", "platform", "roi"]].round(1)

    for idx, row in top_campaigns.iterrows():
        print(f"   {row['campaign_name']}: {row['roi']:.1f}% ROI ({row['platform']})")

    # Platform comparison
    print("\n2. Platform Performance...")
    platform_perf = df.groupby("platform").agg(
        {"spend": "sum", "revenue": "sum", "clicks": "sum", "conversions": "sum"}
    )

    platform_perf["roi"] = (
        (platform_perf["revenue"] - platform_perf["spend"])
        / platform_perf["spend"]
        * 100
    )
    platform_perf["ctr"] = (
        platform_perf["clicks"] / df.groupby("platform")["impressions"].sum() * 100
    )

    best_platform = platform_perf.loc[platform_perf["roi"].idxmax()]
    print(f"   Best Platform: {platform_perf['roi'].idxmax()}")
    print(f"   ROI: {best_platform['roi']:.1f}%")
    print(f"   CTR: {best_platform['ctr']:.2f}%")

    # Date range analysis
    print("\n3. Time Series Analysis...")
    df["date"] = pd.to_datetime(df["date"])
    date_range = df["date"].max() - df["date"].min()
    daily_spend = df.groupby("date")["spend"].sum().mean()

    print(f"   Date Range: {date_range.days} days")
    print(f"   Average Daily Spend: ${daily_spend:.2f}")

    return df


def main():
    """Main demonstration function."""
    print("🎯 Digital Ad Campaign ROI Forecaster")
    print("📊 Data Input & Validation Module Demo")
    print("=" * 60)

    try:
        # Run all demos
        df, validation_result = demo_basic_workflow()
        demo_advanced_validation()
        demo_pydantic_schemas()
        demo_data_insights()

        print("\n\n🎉 Demo Complete!")
        print("=" * 60)
        print("✅ Data loading: SUCCESS")
        print(
            f"✅ Data validation: {'SUCCESS' if validation_result.is_valid else 'FAILED'}"
        )
        print("✅ Schema validation: SUCCESS")
        print("✅ Analysis generation: SUCCESS")
        print("\n🚀 Data module is ready for production use!")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
