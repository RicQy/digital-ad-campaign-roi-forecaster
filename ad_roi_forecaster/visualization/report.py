"""
Report generation module for digital ad campaign ROI forecasting.

This module provides functionality to generate comprehensive PDF and HTML reports
with matplotlib figures, data analysis, and forecasting results.

Main functions:
- generate_pdf_report: Create PDF report with embedded plots
- generate_html_report: Create HTML report with embedded plots
- create_executive_summary: Generate executive summary of findings
- ReportGenerator: Main class for report generation
"""

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import base64
import io
import json
from dataclasses import dataclass

from .plots import (
    plot_spend_vs_conversions,
    plot_forecasted_conversions,
    plot_budget_allocation,
    plot_campaign_performance,
    create_roi_trend_plot,
    save_all_plots
)


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    include_plots: bool = True
    include_summary: bool = True
    include_forecasts: bool = True
    include_optimization: bool = True
    plot_dpi: int = 300
    report_title: str = "Digital Ad Campaign ROI Analysis"
    company_name: str = "Campaign Analytics"
    logo_path: Optional[str] = None


class ReportGenerator:
    """
    Main class for generating comprehensive campaign reports.
    
    This class handles the generation of both PDF and HTML reports with
    embedded plots, data analysis, and forecasting results.
    """
    
    def __init__(self, config: Optional[ReportConfig] = None):
        """
        Initialize the report generator.
        
        Args:
            config: Report configuration options
        """
        self.config = config or ReportConfig()
        self.figures = []
        self.summary_stats = {}
        
    def generate_pdf_report(
        self,
        df: pd.DataFrame,
        output_path: str,
        forecast_data: Optional[pd.DataFrame] = None,
        budget_allocation_data: Optional[Dict] = None,
        additional_info: Optional[Dict] = None
    ) -> str:
        """
        Generate a comprehensive PDF report.
        
        Args:
            df: Campaign data DataFrame
            output_path: Path to save the PDF report
            forecast_data: Optional forecast data
            budget_allocation_data: Optional budget allocation data
            additional_info: Additional information to include in report
            
        Returns:
            Path to the generated PDF file
        """
        # Calculate summary statistics
        self._calculate_summary_stats(df)
        
        # Create PDF
        with pdf_backend.PdfPages(output_path) as pdf_pages:
            # Cover page
            self._create_cover_page(pdf_pages)
            
            # Executive summary
            if self.config.include_summary:
                self._create_executive_summary_page(pdf_pages, df, additional_info)
            
            # Data overview
            self._create_data_overview_page(pdf_pages, df)
            
            # Performance analysis plots
            if self.config.include_plots:
                self._add_performance_plots(pdf_pages, df)
            
            # Forecasting results
            if self.config.include_forecasts and forecast_data is not None:
                self._add_forecast_plots(pdf_pages, df, forecast_data)
            
            # Budget optimization
            if self.config.include_optimization and budget_allocation_data is not None:
                self._add_optimization_plots(pdf_pages, budget_allocation_data)
            
            # Detailed analysis
            self._create_detailed_analysis_page(pdf_pages, df)
            
            # Recommendations
            self._create_recommendations_page(pdf_pages, df)
        
        return output_path
    
    def generate_html_report(
        self,
        df: pd.DataFrame,
        output_path: str,
        forecast_data: Optional[pd.DataFrame] = None,
        budget_allocation_data: Optional[Dict] = None,
        additional_info: Optional[Dict] = None
    ) -> str:
        """
        Generate a comprehensive HTML report.
        
        Args:
            df: Campaign data DataFrame
            output_path: Path to save the HTML report
            forecast_data: Optional forecast data
            budget_allocation_data: Optional budget allocation data
            additional_info: Additional information to include in report
            
        Returns:
            Path to the generated HTML file
        """
        # Calculate summary statistics
        self._calculate_summary_stats(df)
        
        # Generate plots and convert to base64
        plot_images = self._generate_plots_for_html(df, forecast_data, budget_allocation_data)
        
        # Generate HTML content
        html_content = self._create_html_content(df, plot_images, additional_info)
        
        # Save HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    def _calculate_summary_stats(self, df: pd.DataFrame) -> None:
        """Calculate summary statistics for the dataset."""
        df = df.copy()
        
        # Basic metrics
        self.summary_stats = {
            'total_campaigns': len(df),
            'total_spend': df['spend'].sum(),
            'total_revenue': df['revenue'].sum(),
            'total_conversions': df['conversions'].sum(),
            'total_clicks': df['clicks'].sum(),
            'total_impressions': df['impressions'].sum(),
            'date_range': {
                'start': df['date'].min() if 'date' in df.columns else None,
                'end': df['date'].max() if 'date' in df.columns else None
            }
        }
        
        # Calculate derived metrics
        self.summary_stats['overall_roi'] = (
            (self.summary_stats['total_revenue'] - self.summary_stats['total_spend']) / 
            self.summary_stats['total_spend'] * 100
        )
        self.summary_stats['overall_roas'] = (
            self.summary_stats['total_revenue'] / self.summary_stats['total_spend']
        )
        self.summary_stats['overall_ctr'] = (
            self.summary_stats['total_clicks'] / self.summary_stats['total_impressions'] * 100
        )
        self.summary_stats['overall_conversion_rate'] = (
            self.summary_stats['total_conversions'] / self.summary_stats['total_clicks'] * 100
        )
        
        # Platform analysis
        if 'platform' in df.columns:
            platform_stats = df.groupby('platform').agg({
                'spend': 'sum',
                'revenue': 'sum',
                'conversions': 'sum',
                'clicks': 'sum',
                'impressions': 'sum'
            })
            platform_stats['roi'] = (platform_stats['revenue'] - platform_stats['spend']) / platform_stats['spend'] * 100
            self.summary_stats['platform_performance'] = platform_stats.to_dict('index')
        
        # Campaign type analysis
        if 'campaign_type' in df.columns:
            type_stats = df.groupby('campaign_type').agg({
                'spend': 'sum',
                'revenue': 'sum',
                'conversions': 'sum'
            })
            type_stats['roi'] = (type_stats['revenue'] - type_stats['spend']) / type_stats['spend'] * 100
            self.summary_stats['campaign_type_performance'] = type_stats.to_dict('index')
    
    def _create_cover_page(self, pdf_pages: pdf_backend.PdfPages) -> None:
        """Create the cover page for the PDF report."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.8, self.config.report_title, 
                fontsize=24, fontweight='bold', ha='center', va='center')
        
        # Company name
        ax.text(0.5, 0.7, self.config.company_name, 
                fontsize=16, ha='center', va='center')
        
        # Date
        ax.text(0.5, 0.6, f"Generated on: {datetime.now().strftime('%B %d, %Y')}", 
                fontsize=12, ha='center', va='center')
        
        # Summary stats
        if self.summary_stats:
            stats_text = f"""
            Analysis Period: {self.summary_stats['date_range']['start'].strftime('%Y-%m-%d') if self.summary_stats['date_range']['start'] else 'N/A'} to {self.summary_stats['date_range']['end'].strftime('%Y-%m-%d') if self.summary_stats['date_range']['end'] else 'N/A'}
            Total Campaigns: {self.summary_stats['total_campaigns']:,}
            Total Spend: ${self.summary_stats['total_spend']:,.2f}
            Total Revenue: ${self.summary_stats['total_revenue']:,.2f}
            Overall ROI: {self.summary_stats['overall_roi']:.1f}%
            """
            ax.text(0.5, 0.4, stats_text, 
                    fontsize=12, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        pdf_pages.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_executive_summary_page(self, pdf_pages: pdf_backend.PdfPages, df: pd.DataFrame, additional_info: Optional[Dict] = None) -> None:
        """Create the executive summary page."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, "Executive Summary", 
                fontsize=18, fontweight='bold', ha='center', va='top')
        
        # Key findings
        key_findings = self._generate_key_findings(df)
        ax.text(0.1, 0.85, "Key Findings:", 
                fontsize=14, fontweight='bold', ha='left', va='top')
        ax.text(0.1, 0.75, key_findings, 
                fontsize=11, ha='left', va='top', wrap=True)
        
        # Performance metrics
        metrics_text = f"""
        Performance Metrics:
        
        • Total Investment: ${self.summary_stats['total_spend']:,.2f}
        • Total Revenue: ${self.summary_stats['total_revenue']:,.2f}
        • Net Profit: ${self.summary_stats['total_revenue'] - self.summary_stats['total_spend']:,.2f}
        • Return on Investment: {self.summary_stats['overall_roi']:.1f}%
        • Return on Ad Spend: {self.summary_stats['overall_roas']:.2f}x
        • Click-Through Rate: {self.summary_stats['overall_ctr']:.2f}%
        • Conversion Rate: {self.summary_stats['overall_conversion_rate']:.2f}%
        • Total Conversions: {self.summary_stats['total_conversions']:,}
        """
        
        ax.text(0.1, 0.45, metrics_text, 
                fontsize=11, ha='left', va='top')
        
        # Additional information
        if additional_info:
            info_text = "Additional Information:\n"
            for key, value in additional_info.items():
                info_text += f"• {key}: {value}\n"
            ax.text(0.1, 0.15, info_text, 
                    fontsize=11, ha='left', va='top')
        
        pdf_pages.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_data_overview_page(self, pdf_pages: pdf_backend.PdfPages, df: pd.DataFrame) -> None:
        """Create the data overview page."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, "Data Overview", 
                fontsize=18, fontweight='bold', ha='center', va='top')
        
        # Dataset summary
        dataset_info = f"""
        Dataset Summary:
        
        • Total Records: {len(df):,}
        • Date Range: {df['date'].min().strftime('%Y-%m-%d') if 'date' in df.columns else 'N/A'} to {df['date'].max().strftime('%Y-%m-%d') if 'date' in df.columns else 'N/A'}
        • Number of Platforms: {df['platform'].nunique() if 'platform' in df.columns else 'N/A'}
        • Number of Campaign Types: {df['campaign_type'].nunique() if 'campaign_type' in df.columns else 'N/A'}
        • Number of Unique Campaigns: {df['campaign_id'].nunique() if 'campaign_id' in df.columns else 'N/A'}
        """
        
        ax.text(0.1, 0.85, dataset_info, 
                fontsize=11, ha='left', va='top')
        
        # Platform breakdown
        if 'platform' in df.columns and 'platform_performance' in self.summary_stats:
            platform_text = "Platform Performance:\n\n"
            for platform, stats in self.summary_stats['platform_performance'].items():
                platform_text += f"• {platform.title()}: ${stats['spend']:,.0f} spend, ${stats['revenue']:,.0f} revenue, {stats['roi']:.1f}% ROI\n"
            
            ax.text(0.1, 0.6, platform_text, 
                    fontsize=11, ha='left', va='top')
        
        # Campaign type breakdown
        if 'campaign_type' in df.columns and 'campaign_type_performance' in self.summary_stats:
            type_text = "Campaign Type Performance:\n\n"
            for camp_type, stats in self.summary_stats['campaign_type_performance'].items():
                type_text += f"• {camp_type.title()}: ${stats['spend']:,.0f} spend, ${stats['revenue']:,.0f} revenue, {stats['roi']:.1f}% ROI\n"
            
            ax.text(0.1, 0.3, type_text, 
                    fontsize=11, ha='left', va='top')
        
        pdf_pages.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _add_performance_plots(self, pdf_pages: pdf_backend.PdfPages, df: pd.DataFrame) -> None:
        """Add performance analysis plots to the PDF."""
        # Spend vs Conversions plot
        fig = plot_spend_vs_conversions(df, show_plot=False)
        pdf_pages.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Performance dashboard
        fig = plot_campaign_performance(df, show_plot=False)
        pdf_pages.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # ROI trend (if date data available)
        if 'date' in df.columns:
            fig = create_roi_trend_plot(df, show_plot=False)
            pdf_pages.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    def _add_forecast_plots(self, pdf_pages: pdf_backend.PdfPages, df: pd.DataFrame, forecast_data: pd.DataFrame) -> None:
        """Add forecasting plots to the PDF."""
        fig = plot_forecasted_conversions(df, forecast_data, show_plot=False)
        pdf_pages.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _add_optimization_plots(self, pdf_pages: pdf_backend.PdfPages, budget_allocation_data: Dict) -> None:
        """Add optimization plots to the PDF."""
        fig = plot_budget_allocation(
            budget_allocation_data['campaign_names'],
            budget_allocation_data['current_allocation'],
            budget_allocation_data['optimized_allocation'],
            budget_allocation_data['roi_values'],
            show_plot=False
        )
        pdf_pages.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_detailed_analysis_page(self, pdf_pages: pdf_backend.PdfPages, df: pd.DataFrame) -> None:
        """Create detailed analysis page."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, "Detailed Analysis", 
                fontsize=18, fontweight='bold', ha='center', va='top')
        
        # Statistical analysis
        stats_text = self._generate_statistical_analysis(df)
        ax.text(0.1, 0.85, stats_text, 
                fontsize=11, ha='left', va='top')
        
        pdf_pages.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_recommendations_page(self, pdf_pages: pdf_backend.PdfPages, df: pd.DataFrame) -> None:
        """Create recommendations page."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, "Recommendations", 
                fontsize=18, fontweight='bold', ha='center', va='top')
        
        # Generate recommendations
        recommendations = self._generate_recommendations(df)
        ax.text(0.1, 0.85, recommendations, 
                fontsize=11, ha='left', va='top')
        
        pdf_pages.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _generate_plots_for_html(self, df: pd.DataFrame, forecast_data: Optional[pd.DataFrame], budget_allocation_data: Optional[Dict]) -> Dict[str, str]:
        """Generate plots and convert to base64 for HTML embedding."""
        plots = {}
        
        # Spend vs Conversions
        fig = plot_spend_vs_conversions(df, show_plot=False)
        plots['spend_conversions'] = self._fig_to_base64(fig)
        plt.close(fig)
        
        # Performance dashboard
        fig = plot_campaign_performance(df, show_plot=False)
        plots['performance_dashboard'] = self._fig_to_base64(fig)
        plt.close(fig)
        
        # ROI trend
        if 'date' in df.columns:
            fig = create_roi_trend_plot(df, show_plot=False)
            plots['roi_trend'] = self._fig_to_base64(fig)
            plt.close(fig)
        
        # Forecasting plot
        if forecast_data is not None:
            fig = plot_forecasted_conversions(df, forecast_data, show_plot=False)
            plots['forecast'] = self._fig_to_base64(fig)
            plt.close(fig)
        
        # Budget allocation
        if budget_allocation_data is not None:
            fig = plot_budget_allocation(
                budget_allocation_data['campaign_names'],
                budget_allocation_data['current_allocation'],
                budget_allocation_data['optimized_allocation'],
                budget_allocation_data['roi_values'],
                show_plot=False
            )
            plots['budget_allocation'] = self._fig_to_base64(fig)
            plt.close(fig)
        
        return plots
    
    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """Convert matplotlib figure to base64 string."""
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=self.config.plot_dpi, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        return img_str
    
    def _create_html_content(self, df: pd.DataFrame, plot_images: Dict[str, str], additional_info: Optional[Dict] = None) -> str:
        """Create HTML content for the report."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .summary {{ background-color: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                .metrics {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .metric {{ text-align: center; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .metric-label {{ font-size: 14px; color: #7f8c8d; }}
                .plot {{ text-align: center; margin: 30px 0; }}
                .plot img {{ max-width: 100%; height: auto; }}
                .section {{ margin: 40px 0; }}
                .section h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                .recommendations {{ background-color: #e8f6f3; padding: 20px; border-radius: 8px; }}
                .recommendation {{ margin: 10px 0; padding: 10px; background-color: white; border-radius: 4px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{title}</h1>
                <p>{company_name}</p>
                <p>Generated on: {date}</p>
            </div>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value">${total_spend:,.0f}</div>
                        <div class="metric-label">Total Spend</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${total_revenue:,.0f}</div>
                        <div class="metric-label">Total Revenue</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{overall_roi:.1f}%</div>
                        <div class="metric-label">Overall ROI</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{total_conversions:,}</div>
                        <div class="metric-label">Total Conversions</div>
                    </div>
                </div>
                <p>{key_findings}</p>
            </div>
            
            <div class="section">
                <h2>Performance Analysis</h2>
                {performance_plots}
            </div>
            
            <div class="section">
                <h2>Detailed Analysis</h2>
                <p>{detailed_analysis}</p>
            </div>
            
            <div class="recommendations">
                <h2>Recommendations</h2>
                {recommendations}
            </div>
            
            <div class="section">
                <h2>Additional Information</h2>
                <p>Report generated using Digital Ad Campaign ROI Forecaster</p>
                {additional_info}
            </div>
        </body>
        </html>
        """
        
        # Generate performance plots HTML
        performance_plots_html = ""
        for plot_name, plot_base64 in plot_images.items():
            plot_title = plot_name.replace('_', ' ').title()
            performance_plots_html += f"""
            <div class="plot">
                <h3>{plot_title}</h3>
                <img src="data:image/png;base64,{plot_base64}" alt="{plot_title}">
            </div>
            """
        
        # Generate recommendations HTML
        recommendations_list = self._generate_recommendations(df).split('\n')
        recommendations_html = ""
        for rec in recommendations_list:
            if rec.strip():
                recommendations_html += f'<div class="recommendation">{rec.strip()}</div>'
        
        # Additional info HTML
        additional_info_html = ""
        if additional_info:
            for key, value in additional_info.items():
                additional_info_html += f"<p><strong>{key}:</strong> {value}</p>"
        
        return html_template.format(
            title=self.config.report_title,
            company_name=self.config.company_name,
            date=datetime.now().strftime('%B %d, %Y'),
            total_spend=self.summary_stats['total_spend'],
            total_revenue=self.summary_stats['total_revenue'],
            overall_roi=self.summary_stats['overall_roi'],
            total_conversions=self.summary_stats['total_conversions'],
            key_findings=self._generate_key_findings(df),
            performance_plots=performance_plots_html,
            detailed_analysis=self._generate_statistical_analysis(df),
            recommendations=recommendations_html,
            additional_info=additional_info_html
        )
    
    def _generate_key_findings(self, df: pd.DataFrame) -> str:
        """Generate key findings text."""
        findings = []
        
        # ROI analysis
        if self.summary_stats['overall_roi'] > 0:
            findings.append(f"Campaigns generated a positive ROI of {self.summary_stats['overall_roi']:.1f}%")
        else:
            findings.append(f"Campaigns resulted in a negative ROI of {self.summary_stats['overall_roi']:.1f}%")
        
        # Best performing platform
        if 'platform_performance' in self.summary_stats:
            best_platform = max(self.summary_stats['platform_performance'].items(), key=lambda x: x[1]['roi'])
            findings.append(f"Best performing platform: {best_platform[0].title()} with {best_platform[1]['roi']:.1f}% ROI")
        
        # Conversion insights
        if self.summary_stats['overall_conversion_rate'] > 2:
            findings.append("Strong conversion performance with above-average conversion rates")
        elif self.summary_stats['overall_conversion_rate'] < 1:
            findings.append("Conversion rates below industry average, indicating optimization opportunities")
        
        return ". ".join(findings) + "."
    
    def _generate_statistical_analysis(self, df: pd.DataFrame) -> str:
        """Generate statistical analysis text."""
        df_copy = df.copy()
        df_copy['roi'] = (df_copy['revenue'] - df_copy['spend']) / df_copy['spend'] * 100
        
        analysis = f"""
        Statistical Analysis:
        
        • ROI Distribution: Mean {df_copy['roi'].mean():.1f}%, Median {df_copy['roi'].median():.1f}%, Std Dev {df_copy['roi'].std():.1f}%
        • Spend Distribution: Mean ${df_copy['spend'].mean():.2f}, Median ${df_copy['spend'].median():.2f}
        • Revenue Distribution: Mean ${df_copy['revenue'].mean():.2f}, Median ${df_copy['revenue'].median():.2f}
        • Conversion Rate: Mean {(df_copy['conversions'] / df_copy['clicks']).mean() * 100:.2f}%
        • Click-Through Rate: Mean {(df_copy['clicks'] / df_copy['impressions']).mean() * 100:.2f}%
        • Correlation between Spend and Conversions: {df_copy['spend'].corr(df_copy['conversions']):.3f}
        • Correlation between Spend and Revenue: {df_copy['spend'].corr(df_copy['revenue']):.3f}
        """
        
        return analysis
    
    def _generate_recommendations(self, df: pd.DataFrame) -> str:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # ROI-based recommendations
        if self.summary_stats['overall_roi'] < 0:
            recommendations.append("• Focus on improving ROI by reducing spend on underperforming campaigns")
            recommendations.append("• Analyze high-performing campaigns and replicate their strategies")
        elif self.summary_stats['overall_roi'] > 50:
            recommendations.append("• Consider increasing budget allocation to high-performing campaigns")
            recommendations.append("• Explore scaling successful campaigns to new markets or audiences")
        
        # Platform recommendations
        if 'platform_performance' in self.summary_stats:
            platform_rois = {k: v['roi'] for k, v in self.summary_stats['platform_performance'].items()}
            best_platform = max(platform_rois, key=platform_rois.get)
            worst_platform = min(platform_rois, key=platform_rois.get)
            
            recommendations.append(f"• Increase investment in {best_platform.title()} campaigns (highest ROI)")
            if platform_rois[worst_platform] < 0:
                recommendations.append(f"• Consider reducing or optimizing {worst_platform.title()} campaigns (negative ROI)")
        
        # Conversion rate recommendations
        if self.summary_stats['overall_conversion_rate'] < 1:
            recommendations.append("• Improve landing page experience to increase conversion rates")
            recommendations.append("• Review and optimize targeting parameters")
        
        # Budget allocation recommendations
        recommendations.append("• Implement automated bid adjustments based on performance data")
        recommendations.append("• Set up regular performance monitoring and optimization reviews")
        
        return "\n".join(recommendations)


# Convenience functions
def generate_pdf_report(
    df: pd.DataFrame,
    output_path: str,
    config: Optional[ReportConfig] = None,
    forecast_data: Optional[pd.DataFrame] = None,
    budget_allocation_data: Optional[Dict] = None,
    additional_info: Optional[Dict] = None
) -> str:
    """
    Generate a PDF report with campaign analysis.
    
    Args:
        df: Campaign data DataFrame
        output_path: Path to save the PDF report
        config: Report configuration
        forecast_data: Optional forecast data
        budget_allocation_data: Optional budget allocation data
        additional_info: Additional information to include
        
    Returns:
        Path to the generated PDF file
    """
    generator = ReportGenerator(config)
    return generator.generate_pdf_report(df, output_path, forecast_data, budget_allocation_data, additional_info)


def generate_html_report(
    df: pd.DataFrame,
    output_path: str,
    config: Optional[ReportConfig] = None,
    forecast_data: Optional[pd.DataFrame] = None,
    budget_allocation_data: Optional[Dict] = None,
    additional_info: Optional[Dict] = None
) -> str:
    """
    Generate an HTML report with campaign analysis.
    
    Args:
        df: Campaign data DataFrame
        output_path: Path to save the HTML report
        config: Report configuration
        forecast_data: Optional forecast data
        budget_allocation_data: Optional budget allocation data
        additional_info: Additional information to include
        
    Returns:
        Path to the generated HTML file
    """
    generator = ReportGenerator(config)
    return generator.generate_html_report(df, output_path, forecast_data, budget_allocation_data, additional_info)


def create_executive_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create an executive summary of campaign performance.
    
    Args:
        df: Campaign data DataFrame
        
    Returns:
        Dictionary containing executive summary data
    """
    generator = ReportGenerator()
    generator._calculate_summary_stats(df)
    
    return {
        'summary_stats': generator.summary_stats,
        'key_findings': generator._generate_key_findings(df),
        'recommendations': generator._generate_recommendations(df),
        'statistical_analysis': generator._generate_statistical_analysis(df)
    }
