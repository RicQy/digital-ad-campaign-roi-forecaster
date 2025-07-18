#!/usr/bin/env python3
"""
Visualization & Reporting Module Demo

This script demonstrates the complete functionality of the visualization and reporting module:
1. Loading sample campaign data
2. Generating various matplotlib/seaborn plots
3. Creating forecasting visualizations
4. Generating budget allocation charts
5. Creating comprehensive PDF and HTML reports
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ad_roi_forecaster.data import load_campaign_data
from ad_roi_forecaster.visualization import (
    plot_spend_vs_conversions,
    plot_forecasted_conversions,
    plot_budget_allocation,
    plot_campaign_performance,
    create_roi_trend_plot,
    save_all_plots,
    generate_pdf_report,
    generate_html_report,
    create_executive_summary,
    ReportConfig
)


def demo_basic_plots():
    """Demonstrate basic plotting functionality."""
    print("🎨 Basic Plotting Demo")
    print("=" * 50)
    
    # Load sample data
    print("\n1. Loading Sample Data...")
    csv_path = project_root / "sample_data" / "campaign_data.csv"
    
    if not csv_path.exists():
        print(f"   ⚠ Sample data not found at {csv_path}")
        print("   Creating synthetic data for demo...")
        df = create_synthetic_data()
    else:
        df = load_campaign_data(csv_path)
    
    print(f"   ✓ Loaded {len(df)} records")
    
    # Create output directory
    output_dir = project_root / "demo_output"
    output_dir.mkdir(exist_ok=True)
    
    # Plot 1: Spend vs Conversions
    print("\n2. Creating Spend vs Conversions Plot...")
    fig = plot_spend_vs_conversions(
        df, 
        save_path=str(output_dir / "spend_vs_conversions.png"),
        show_plot=False
    )
    print("   ✓ Spend vs Conversions plot saved")
    
    # Plot 2: Campaign Performance Dashboard
    print("\n3. Creating Performance Dashboard...")
    fig = plot_campaign_performance(
        df, 
        save_path=str(output_dir / "performance_dashboard.png"),
        show_plot=False
    )
    print("   ✓ Performance dashboard saved")
    
    # Plot 3: ROI Trend Analysis
    print("\n4. Creating ROI Trend Plot...")
    fig = create_roi_trend_plot(
        df, 
        save_path=str(output_dir / "roi_trend.png"),
        show_plot=False
    )
    print("   ✓ ROI trend plot saved")
    
    return df


def demo_forecasting_plots():
    """Demonstrate forecasting visualization."""
    print("\n\n🔮 Forecasting Visualization Demo")
    print("=" * 50)
    
    # Load data
    csv_path = project_root / "sample_data" / "campaign_data.csv"
    if not csv_path.exists():
        df = create_synthetic_data()
    else:
        df = load_campaign_data(csv_path)
    
    # Create synthetic forecast data
    print("\n1. Creating Synthetic Forecast Data...")
    forecast_data = create_synthetic_forecast_data(df)
    print(f"   ✓ Created {len(forecast_data)} forecast points")
    
    # Create output directory
    output_dir = project_root / "demo_output"
    output_dir.mkdir(exist_ok=True)
    
    # Plot forecasted conversions
    print("\n2. Creating Forecasted Conversions Plot...")
    fig = plot_forecasted_conversions(
        df,
        forecast_data,
        save_path=str(output_dir / "forecasted_conversions.png"),
        show_plot=False
    )
    print("   ✓ Forecasted conversions plot saved")
    
    return df, forecast_data


def demo_budget_allocation_plots():
    """Demonstrate budget allocation visualization."""
    print("\n\n💰 Budget Allocation Visualization Demo")
    print("=" * 50)
    
    # Create synthetic budget allocation data
    print("\n1. Creating Synthetic Budget Allocation Data...")
    budget_data = create_synthetic_budget_data()
    print(f"   ✓ Created budget data for {len(budget_data['campaign_names'])} campaigns")
    
    # Create output directory
    output_dir = project_root / "demo_output"
    output_dir.mkdir(exist_ok=True)
    
    # Plot budget allocation
    print("\n2. Creating Budget Allocation Plot...")
    fig = plot_budget_allocation(
        budget_data['campaign_names'],
        budget_data['current_allocation'],
        budget_data['optimized_allocation'],
        budget_data['roi_values'],
        save_path=str(output_dir / "budget_allocation.png"),
        show_plot=False
    )
    print("   ✓ Budget allocation plot saved")
    
    return budget_data


def demo_comprehensive_plots():
    """Demonstrate comprehensive plot generation."""
    print("\n\n📊 Comprehensive Plots Demo")
    print("=" * 50)
    
    # Load data
    csv_path = project_root / "sample_data" / "campaign_data.csv"
    if not csv_path.exists():
        df = create_synthetic_data()
    else:
        df = load_campaign_data(csv_path)
    
    # Create synthetic forecast and budget data
    forecast_data = create_synthetic_forecast_data(df)
    budget_data = create_synthetic_budget_data()
    
    # Create output directory
    output_dir = project_root / "demo_output"
    output_dir.mkdir(exist_ok=True)
    
    # Generate all plots
    print("\n1. Generating All Plots...")
    saved_files = save_all_plots(
        df,
        str(output_dir),
        forecast_data=forecast_data,
        budget_allocation_data=budget_data,
        prefix="comprehensive_demo"
    )
    
    print(f"   ✓ Generated {len(saved_files)} plots:")
    for file_path in saved_files:
        print(f"     - {Path(file_path).name}")
    
    return df, forecast_data, budget_data


def demo_pdf_report():
    """Demonstrate PDF report generation."""
    print("\n\n📄 PDF Report Generation Demo")
    print("=" * 50)
    
    # Load data
    csv_path = project_root / "sample_data" / "campaign_data.csv"
    if not csv_path.exists():
        df = create_synthetic_data()
    else:
        df = load_campaign_data(csv_path)
    
    # Create synthetic forecast and budget data
    forecast_data = create_synthetic_forecast_data(df)
    budget_data = create_synthetic_budget_data()
    
    # Create output directory
    output_dir = project_root / "demo_output"
    output_dir.mkdir(exist_ok=True)
    
    # Configure report
    config = ReportConfig(
        report_title="Campaign Performance Analysis Report",
        company_name="Demo Analytics Inc.",
        include_plots=True,
        include_summary=True,
        include_forecasts=True,
        include_optimization=True
    )
    
    # Additional information
    additional_info = {
        "Analysis Type": "Comprehensive Campaign Review",
        "Report Version": "1.0",
        "Analyst": "Demo User",
        "Business Unit": "Digital Marketing"
    }
    
    # Generate PDF report
    print("\n1. Generating PDF Report...")
    pdf_path = str(output_dir / "campaign_analysis_report.pdf")
    
    generated_path = generate_pdf_report(
        df,
        pdf_path,
        config=config,
        forecast_data=forecast_data,
        budget_allocation_data=budget_data,
        additional_info=additional_info
    )
    
    print(f"   ✓ PDF report generated: {Path(generated_path).name}")
    print(f"   📁 Location: {generated_path}")
    
    return generated_path


def demo_html_report():
    """Demonstrate HTML report generation."""
    print("\n\n🌐 HTML Report Generation Demo")
    print("=" * 50)
    
    # Load data
    csv_path = project_root / "sample_data" / "campaign_data.csv"
    if not csv_path.exists():
        df = create_synthetic_data()
    else:
        df = load_campaign_data(csv_path)
    
    # Create synthetic forecast and budget data
    forecast_data = create_synthetic_forecast_data(df)
    budget_data = create_synthetic_budget_data()
    
    # Create output directory
    output_dir = project_root / "demo_output"
    output_dir.mkdir(exist_ok=True)
    
    # Configure report
    config = ReportConfig(
        report_title="Interactive Campaign Analysis Dashboard",
        company_name="Demo Analytics Inc.",
        include_plots=True,
        include_summary=True,
        include_forecasts=True,
        include_optimization=True
    )
    
    # Additional information
    additional_info = {
        "Analysis Type": "Interactive Dashboard",
        "Report Version": "1.0",
        "Analyst": "Demo User",
        "Business Unit": "Digital Marketing",
        "Dashboard Features": "Interactive charts and detailed analytics"
    }
    
    # Generate HTML report
    print("\n1. Generating HTML Report...")
    html_path = str(output_dir / "campaign_analysis_dashboard.html")
    
    generated_path = generate_html_report(
        df,
        html_path,
        config=config,
        forecast_data=forecast_data,
        budget_allocation_data=budget_data,
        additional_info=additional_info
    )
    
    print(f"   ✓ HTML report generated: {Path(generated_path).name}")
    print(f"   📁 Location: {generated_path}")
    print(f"   🌐 Open in browser: file://{generated_path}")
    
    return generated_path


def demo_executive_summary():
    """Demonstrate executive summary generation."""
    print("\n\n📋 Executive Summary Demo")
    print("=" * 50)
    
    # Load data
    csv_path = project_root / "sample_data" / "campaign_data.csv"
    if not csv_path.exists():
        df = create_synthetic_data()
    else:
        df = load_campaign_data(csv_path)
    
    # Generate executive summary
    print("\n1. Generating Executive Summary...")
    summary = create_executive_summary(df)
    
    # Display key metrics
    print("\n2. Key Performance Metrics:")
    stats = summary['summary_stats']
    print(f"   • Total Spend: ${stats['total_spend']:,.2f}")
    print(f"   • Total Revenue: ${stats['total_revenue']:,.2f}")
    print(f"   • Overall ROI: {stats['overall_roi']:.1f}%")
    print(f"   • Total Conversions: {stats['total_conversions']:,}")
    print(f"   • Overall CTR: {stats['overall_ctr']:.2f}%")
    
    # Display key findings
    print("\n3. Key Findings:")
    print(f"   {summary['key_findings']}")
    
    # Display top recommendations
    print("\n4. Top Recommendations:")
    recommendations = summary['recommendations'].split('\n')[:3]
    for rec in recommendations:
        if rec.strip():
            print(f"   {rec.strip()}")
    
    return summary


def create_synthetic_data():
    """Create synthetic campaign data for demo purposes."""
    np.random.seed(42)
    
    # Generate 100 campaign records
    n_records = 100
    
    # Date range
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 6, 30)
    dates = pd.date_range(start=start_date, end=end_date, periods=n_records)
    
    # Platforms and campaign types
    platforms = ['google', 'facebook', 'instagram', 'linkedin', 'twitter']
    campaign_types = ['search', 'display', 'video', 'social', 'shopping']
    
    # Generate synthetic data
    data = []
    for i in range(n_records):
        # Base metrics
        spend = np.random.uniform(100, 5000)
        impressions = int(np.random.uniform(1000, 50000))
        clicks = int(impressions * np.random.uniform(0.01, 0.05))  # 1-5% CTR
        conversions = int(clicks * np.random.uniform(0.01, 0.1))  # 1-10% conversion rate
        
        # Revenue with some correlation to spend and conversions
        revenue = spend * np.random.uniform(0.8, 2.5) + conversions * np.random.uniform(10, 100)
        
        record = {
            'campaign_id': f'CAMP_{i+1:03d}',
            'campaign_name': f'Campaign {i+1}',
            'campaign_type': np.random.choice(campaign_types),
            'platform': np.random.choice(platforms),
            'date': dates[i].strftime('%Y-%m-%d'),
            'spend': round(spend, 2),
            'impressions': impressions,
            'clicks': clicks,
            'conversions': conversions,
            'revenue': round(revenue, 2),
            'target_audience': f'Audience_{np.random.randint(1, 6)}',
            'geographic_region': np.random.choice(['US', 'UK', 'CA', 'AU', 'DE'])
        }
        
        data.append(record)
    
    return pd.DataFrame(data)


def create_synthetic_forecast_data(df):
    """Create synthetic forecast data."""
    # Get date range from historical data
    df['date'] = pd.to_datetime(df['date'])
    last_date = df['date'].max()
    
    # Create 30 days of forecast data
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30, freq='D')
    
    # Generate forecast conversions with some trend and seasonality
    base_conversions = df['conversions'].mean()
    trend = np.linspace(0, 10, 30)  # Small upward trend
    seasonality = 5 * np.sin(np.arange(30) * 2 * np.pi / 7)  # Weekly seasonality
    noise = np.random.normal(0, 5, 30)
    
    forecast_conversions = base_conversions + trend + seasonality + noise
    forecast_conversions = np.maximum(forecast_conversions, 0)  # Ensure positive
    
    forecast_data = pd.DataFrame({
        'date': forecast_dates,
        'conversions': forecast_conversions.round().astype(int)
    })
    
    return forecast_data


def create_synthetic_budget_data():
    """Create synthetic budget allocation data."""
    campaigns = ['Search Campaign A', 'Display Campaign B', 'Social Campaign C', 'Video Campaign D', 'Shopping Campaign E']
    
    # Current allocation
    current_allocation = [15000, 12000, 8000, 10000, 5000]
    
    # Optimized allocation (with some changes)
    optimized_allocation = [18000, 10000, 9000, 8000, 5000]
    
    # ROI values for each campaign
    roi_values = [45.2, 23.8, 67.1, 12.5, 89.3]
    
    return {
        'campaign_names': campaigns,
        'current_allocation': current_allocation,
        'optimized_allocation': optimized_allocation,
        'roi_values': roi_values
    }


def main():
    """Main demonstration function."""
    print("🎯 Digital Ad Campaign ROI Forecaster")
    print("📊 Visualization & Reporting Module Demo")
    print("=" * 60)
    
    try:
        # Create demo output directory
        output_dir = project_root / "demo_output"
        output_dir.mkdir(exist_ok=True)
        
        # Run all demos
        print("\n🚀 Starting Comprehensive Demo...")
        
        # 1. Basic plotting
        df = demo_basic_plots()
        
        # 2. Forecasting plots
        df, forecast_data = demo_forecasting_plots()
        
        # 3. Budget allocation plots
        budget_data = demo_budget_allocation_plots()
        
        # 4. Comprehensive plots
        df, forecast_data, budget_data = demo_comprehensive_plots()
        
        # 5. PDF report
        pdf_path = demo_pdf_report()
        
        # 6. HTML report
        html_path = demo_html_report()
        
        # 7. Executive summary
        summary = demo_executive_summary()
        
        # Final summary
        print("\n\n🎉 Demo Complete!")
        print("=" * 60)
        print("✅ Basic plotting: SUCCESS")
        print("✅ Forecasting visualization: SUCCESS")
        print("✅ Budget allocation charts: SUCCESS")
        print("✅ Comprehensive plots: SUCCESS")
        print("✅ PDF report generation: SUCCESS")
        print("✅ HTML report generation: SUCCESS")
        print("✅ Executive summary: SUCCESS")
        
        print(f"\n📁 All outputs saved to: {output_dir}")
        print(f"📄 PDF Report: {Path(pdf_path).name}")
        print(f"🌐 HTML Report: {Path(html_path).name}")
        print("\n🚀 Visualization & Reporting module is ready for production use!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
