"""
Visualization & Reporting module for digital ad campaign ROI forecasting.

This module provides comprehensive visualization and reporting capabilities
for campaign data analysis, forecasting results, and optimization outcomes.

Main components:
- plots: matplotlib/seaborn charts for data visualization
- report: PDF/HTML report generation with embedded plots

Visualization functions:
- plot_spend_vs_conversions: Scatter plot with regression line
- plot_forecasted_conversions: Time series forecasting visualization
- plot_budget_allocation: Bar chart of optimized budget allocation & ROI
- plot_campaign_performance: Comprehensive performance dashboard
- create_roi_trend_plot: ROI trend analysis over time
- save_all_plots: Generate and save all plots

Reporting functions:
- generate_pdf_report: Create comprehensive PDF reports
- generate_html_report: Create interactive HTML reports
- create_executive_summary: Generate executive summary data
- ReportGenerator: Main class for report generation
- ReportConfig: Configuration class for reports
"""

# Import plotting functions
from .plots import (
    plot_spend_vs_conversions,
    plot_forecasted_conversions,
    plot_budget_allocation,
    plot_campaign_performance,
    create_roi_trend_plot,
    save_all_plots
)

# Import reporting functions
from .report import (
    generate_pdf_report,
    generate_html_report,
    create_executive_summary,
    ReportGenerator,
    ReportConfig
)

# Export all main components
__all__ = [
    # Plotting functions
    'plot_spend_vs_conversions',
    'plot_forecasted_conversions',
    'plot_budget_allocation',
    'plot_campaign_performance',
    'create_roi_trend_plot',
    'save_all_plots',
    
    # Reporting functions
    'generate_pdf_report',
    'generate_html_report',
    'create_executive_summary',
    'ReportGenerator',
    'ReportConfig'
]
