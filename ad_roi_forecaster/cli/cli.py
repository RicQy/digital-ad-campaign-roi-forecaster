#!/usr/bin/env python3
"""
CLI entry-point for Digital Ad Campaign ROI Forecaster.

This module provides command-line interface for:
- forecast: Generate forecasts from CSV data
- optimize: Optimize budget allocation
- report: Generate visual reports

Usage:
    python -m ad_roi_forecaster.cli.cli forecast --input data.csv --output forecasts.csv
    python -m ad_roi_forecaster.cli.cli optimize --budget 10000 --start-date 2024-01-01 --end-date 2024-01-31
    python -m ad_roi_forecaster.cli.cli report --input data.csv --output-dir reports/
"""

import click
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sys
from typing import Optional, Dict, Any, List

# Rich for better error handling and logging
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install
from rich.table import Table
from rich.progress import Progress, TaskID

# Install rich traceback handler
install(show_locals=True)

# Project imports
from ad_roi_forecaster.data.loader import load_campaign_data
from ad_roi_forecaster.forecasting.model_selector import forecast_with_best_model
from ad_roi_forecaster.optimization.roi_optimizer import optimize_roi
from ad_roi_forecaster.visualization.report import (
    generate_pdf_report,
    generate_html_report,
)
from ad_roi_forecaster.data.schemas import CampaignDataset, CampaignRecord

# Initialize console for rich output
console = Console()

# Configure logging with rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(level)

    # Set other loggers to WARNING to reduce noise
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose: bool):
    """Digital Ad Campaign ROI Forecaster CLI.

    A comprehensive tool for forecasting and optimizing digital advertising campaign ROI.
    """
    setup_logging(verbose)


@cli.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to input CSV file containing campaign data",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    required=True,
    type=click.Path(),
    help="Path to output CSV file for forecasts",
)
@click.option(
    "--periods",
    "-p",
    default=30,
    type=int,
    help="Number of periods to forecast (default: 30)",
)
@click.option(
    "--spend-scenario",
    "-s",
    type=float,
    help="Spend scenario for forecasting (default: use historical average)",
)
@click.option(
    "--model",
    "-m",
    type=click.Choice(["auto", "baseline", "arima", "seasonal_naive"]),
    default="auto",
    help="Model type to use for forecasting",
)
@click.option(
    "--confidence",
    "-c",
    default=0.95,
    type=float,
    help="Confidence level for predictions (default: 0.95)",
)
@click.option(
    "--cv-folds",
    default=5,
    type=int,
    help="Cross-validation folds for model selection (default: 5)",
)
@click.option(
    "--scoring",
    default="mae",
    type=click.Choice(["mae", "mse", "rmse", "mape", "r2_score"]),
    help="Scoring metric for model selection (default: mae)",
)
def forecast(
    input_path: str,
    output_path: str,
    periods: int,
    spend_scenario: Optional[float],
    model: str,
    confidence: float,
    cv_folds: int,
    scoring: str,
):
    """Generate forecasts from input CSV data.

    This command loads campaign data from a CSV file, selects the best forecasting model,
    and generates predictions for the specified number of periods.

    Example:
        forecast -i data.csv -o forecasts.csv -p 30 -s 1000
    """
    try:
        console.print(f"🚀 Starting forecast generation...", style="bold blue")

        # Load data
        console.print(f"📊 Loading data from {input_path}...")
        df = load_campaign_data(input_path)
        console.print(f"✅ Loaded {len(df)} campaign records", style="green")

        # Display data summary
        _display_data_summary(df)

        # Prepare models based on selection
        models = None
        if model != "auto":
            models = _get_specific_model_config(model, confidence)

        # Generate forecasts
        console.print(f"🔮 Generating forecasts for {periods} periods...")

        with Progress() as progress:
            task = progress.add_task("Forecasting...", total=100)

            progress.update(task, advance=20)
            forecasts, model_info = forecast_with_best_model(
                df=df,
                future_periods=periods,
                spend_scenario=spend_scenario,
                models=models,
                cv_folds=cv_folds,
                scoring=scoring,
                confidence_level=confidence,
            )
            progress.update(task, advance=80)

            # Save forecasts
            forecasts.to_csv(output_path, index=False)
            progress.update(task, advance=100)

        console.print(f"✅ Forecasts saved to {output_path}", style="green")

        # Display model info
        _display_model_info(model_info)

        # Display forecast summary
        _display_forecast_summary(forecasts)

    except Exception as e:
        console.print(f"❌ Error generating forecasts: {str(e)}", style="red")
        logger.exception("Forecast generation failed")
        sys.exit(1)


@cli.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True),
    help="Path to input CSV file containing campaign data (optional)",
)
@click.option(
    "--budget", "-b", required=True, type=float, help="Total budget for optimization"
)
@click.option(
    "--start-date",
    "-s",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Start date for optimization period",
)
@click.option(
    "--end-date",
    "-e",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="End date for optimization period",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(),
    help="Path to output CSV file for allocation results",
)
@click.option(
    "--campaigns",
    "-c",
    multiple=True,
    help="Specific campaign IDs to optimize (can be used multiple times)",
)
@click.option(
    "--platforms",
    "-p",
    multiple=True,
    help="Specific platforms to optimize (can be used multiple times)",
)
def optimize(
    input_path: Optional[str],
    budget: float,
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    output_path: Optional[str],
    campaigns: tuple,
    platforms: tuple,
):
    """Optimize budget allocation for maximum ROI.

    This command optimizes budget allocation across campaigns to maximize ROI.
    Can work with historical data or synthetic scenarios.

    Example:
        optimize -b 10000 -s 2024-01-01 -e 2024-01-31 -i data.csv -o allocation.csv
    """
    try:
        console.print(f"🎯 Starting budget optimization...", style="bold blue")
        console.print(f"💰 Total budget: ${budget:,.2f}")

        # Load data if provided
        df = None
        if input_path:
            console.print(f"📊 Loading data from {input_path}...")
            df = load_campaign_data(input_path)
            console.print(f"✅ Loaded {len(df)} campaign records", style="green")

            # Filter by date range if provided
            if start_date and end_date:
                df["date"] = pd.to_datetime(df["date"])
                df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
                console.print(
                    f"📅 Filtered to {len(df)} records in date range", style="yellow"
                )

            # Filter by campaigns if provided
            if campaigns:
                df = df[df["campaign_id"].isin(campaigns)]
                console.print(
                    f"🎪 Filtered to {len(df)} records for specified campaigns",
                    style="yellow",
                )

            # Filter by platforms if provided
            if platforms:
                df = df[df["platform"].isin(platforms)]
                console.print(
                    f"🌐 Filtered to {len(df)} records for specified platforms",
                    style="yellow",
                )

            if df.empty:
                raise ValueError("No data remaining after filtering")

        else:
            # Create synthetic data for demonstration
            console.print(
                "🔧 No input data provided, creating synthetic campaign data..."
            )
            df = _create_synthetic_campaign_data(budget, start_date, end_date)

        # Display data summary
        _display_data_summary(df)

        # Convert to CampaignDataset for optimization
        console.print("🔄 Converting data for optimization...")
        campaign_dataset = _df_to_campaign_dataset(df)

        # Perform optimization
        console.print("⚡ Optimizing budget allocation...")

        with Progress() as progress:
            task = progress.add_task("Optimizing...", total=100)

            progress.update(task, advance=30)
            optimal_allocation, expected_roi = optimize_roi(campaign_dataset, budget)
            progress.update(task, advance=70)

            # Create results
            results = _create_optimization_results(
                df, optimal_allocation, expected_roi, budget
            )
            progress.update(task, advance=100)

        # Display optimization results
        _display_optimization_results(results, expected_roi)

        # Save results if output path provided
        if output_path:
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_path, index=False)
            console.print(
                f"✅ Optimization results saved to {output_path}", style="green"
            )

    except Exception as e:
        console.print(f"❌ Error during optimization: {str(e)}", style="red")
        logger.exception("Optimization failed")
        sys.exit(1)


@cli.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to input CSV file containing campaign data",
)
@click.option(
    "--output-dir",
    "-o",
    "output_dir",
    required=True,
    type=click.Path(),
    help="Directory to save generated reports",
)
@click.option(
    "--format",
    "-f",
    "report_format",
    multiple=True,
    type=click.Choice(["html", "pdf"]),
    default=["html"],
    help="Report format(s) to generate (default: html)",
)
@click.option(
    "--title", "-t", default="Digital Ad Campaign ROI Analysis", help="Report title"
)
@click.option(
    "--company", "-c", default="Campaign Analytics", help="Company name for report"
)
@click.option(
    "--include-forecasts", is_flag=True, help="Include forecasting analysis in report"
)
@click.option(
    "--forecast-periods",
    default=30,
    type=int,
    help="Number of periods to forecast for report (default: 30)",
)
@click.option(
    "--include-optimization",
    is_flag=True,
    help="Include budget optimization analysis in report",
)
@click.option(
    "--optimization-budget",
    type=float,
    help="Budget for optimization analysis (default: use total historical spend)",
)
@click.option("--dpi", default=300, type=int, help="DPI for plot images (default: 300)")
def report(
    input_path: str,
    output_dir: str,
    report_format: tuple,
    title: str,
    company: str,
    include_forecasts: bool,
    forecast_periods: int,
    include_optimization: bool,
    optimization_budget: Optional[float],
    dpi: int,
):
    """Generate visual reports with charts and analysis.

    This command creates comprehensive reports with visualizations, statistics,
    and insights from campaign data.

    Example:
        report -i data.csv -o reports/ -f html -f pdf --include-forecasts
    """
    try:
        console.print(f"📊 Starting report generation...", style="bold blue")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load data
        console.print(f"📥 Loading data from {input_path}...")
        df = load_campaign_data(input_path)
        console.print(f"✅ Loaded {len(df)} campaign records", style="green")

        # Display data summary
        _display_data_summary(df)

        # Prepare report configuration
        from ad_roi_forecaster.visualization.report import ReportConfig

        config = ReportConfig(
            report_title=title,
            company_name=company,
            plot_dpi=dpi,
            include_forecasts=include_forecasts,
            include_optimization=include_optimization,
        )

        forecast_data = None
        budget_allocation_data = None

        with Progress() as progress:
            total_tasks = (
                len(report_format)
                + (1 if include_forecasts else 0)
                + (1 if include_optimization else 0)
            )
            main_task = progress.add_task("Generating reports...", total=total_tasks)

            # Generate forecasts if requested
            if include_forecasts:
                console.print("🔮 Generating forecasts for report...")
                forecasts, _ = forecast_with_best_model(
                    df=df, future_periods=forecast_periods, confidence_level=0.95
                )
                forecast_data = forecasts
                progress.update(main_task, advance=1)

            # Generate optimization if requested
            if include_optimization:
                console.print("⚡ Performing optimization for report...")
                budget = optimization_budget or df["spend"].sum()
                campaign_dataset = _df_to_campaign_dataset(df)
                optimal_allocation, expected_roi = optimize_roi(
                    campaign_dataset, budget
                )

                budget_allocation_data = {
                    "campaign_names": df["campaign_id"].unique().tolist(),
                    "current_allocation": df.groupby("campaign_id")["spend"]
                    .sum()
                    .tolist(),
                    "optimized_allocation": optimal_allocation.tolist(),
                    "roi_values": [
                        (row["revenue"] - row["spend"]) / row["spend"] * 100
                        for _, row in df.groupby("campaign_id")
                        .agg({"spend": "sum", "revenue": "sum"})
                        .iterrows()
                    ],
                }
                progress.update(main_task, advance=1)

            # Generate reports in requested formats
            for fmt in report_format:
                console.print(f"📄 Generating {fmt.upper()} report...")

                if fmt == "html":
                    output_file = (
                        output_path
                        / f"campaign_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                    )
                    generate_html_report(
                        df=df,
                        output_path=str(output_file),
                        config=config,
                        forecast_data=forecast_data,
                        budget_allocation_data=budget_allocation_data,
                    )
                elif fmt == "pdf":
                    output_file = (
                        output_path
                        / f"campaign_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    )
                    generate_pdf_report(
                        df=df,
                        output_path=str(output_file),
                        config=config,
                        forecast_data=forecast_data,
                        budget_allocation_data=budget_allocation_data,
                    )

                console.print(
                    f"✅ {fmt.upper()} report saved to {output_file}", style="green"
                )
                progress.update(main_task, advance=1)

        console.print(
            f"🎉 All reports generated successfully in {output_dir}", style="bold green"
        )

    except Exception as e:
        console.print(f"❌ Error generating reports: {str(e)}", style="red")
        logger.exception("Report generation failed")
        sys.exit(1)


# Helper functions
def _display_data_summary(df: pd.DataFrame) -> None:
    """Display a summary of the loaded data."""
    table = Table(title="Data Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Total Records", f"{len(df):,}")
    table.add_row(
        "Date Range",
        f"{df['date'].min()} to {df['date'].max()}" if "date" in df.columns else "N/A",
    )
    table.add_row("Total Spend", f"${df['spend'].sum():,.2f}")
    table.add_row("Total Revenue", f"${df['revenue'].sum():,.2f}")
    table.add_row("Total Conversions", f"{df['conversions'].sum():,}")
    table.add_row(
        "Unique Campaigns",
        f"{df['campaign_id'].nunique():,}" if "campaign_id" in df.columns else "N/A",
    )
    table.add_row(
        "Platforms",
        f"{df['platform'].nunique():,}" if "platform" in df.columns else "N/A",
    )

    # Calculate overall ROI
    overall_roi = (
        ((df["revenue"].sum() - df["spend"].sum()) / df["spend"].sum() * 100)
        if df["spend"].sum() > 0
        else 0
    )
    table.add_row("Overall ROI", f"{overall_roi:.1f}%")

    console.print(table)


def _display_model_info(model_info: Dict[str, Any]) -> None:
    """Display information about the selected model."""
    table = Table(title="Model Selection Results")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Selected Model", model_info.get("name", "N/A"))
    table.add_row("Model Type", model_info.get("model_type", "N/A"))
    table.add_row("Scoring Metric", model_info.get("scoring_metric", "N/A"))
    table.add_row("Score", f"{model_info.get('score', 0):.4f}")

    console.print(table)


def _display_forecast_summary(forecasts: pd.DataFrame) -> None:
    """Display a summary of the forecasts."""
    table = Table(title="Forecast Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    if "predicted_conversions" in forecasts.columns:
        table.add_row("Forecast Periods", f"{len(forecasts):,}")
        table.add_row(
            "Total Predicted Conversions",
            f"{forecasts['predicted_conversions'].sum():.0f}",
        )
        table.add_row(
            "Average Daily Conversions",
            f"{forecasts['predicted_conversions'].mean():.1f}",
        )
        table.add_row(
            "Min Prediction", f"{forecasts['predicted_conversions'].min():.1f}"
        )
        table.add_row(
            "Max Prediction", f"{forecasts['predicted_conversions'].max():.1f}"
        )

    console.print(table)


def _display_optimization_results(results: List[Dict], expected_roi: float) -> None:
    """Display optimization results."""
    table = Table(title="Budget Optimization Results")
    table.add_column("Campaign", style="cyan")
    table.add_column("Current Spend", style="yellow")
    table.add_column("Optimized Spend", style="green")
    table.add_column("Change", style="magenta")

    for result in results:
        change = result["optimized_spend"] - result["current_spend"]
        change_pct = (
            (change / result["current_spend"] * 100)
            if result["current_spend"] > 0
            else 0
        )
        change_str = f"{change:+.2f} ({change_pct:+.1f}%)"

        table.add_row(
            result["campaign_id"],
            f"${result['current_spend']:.2f}",
            f"${result['optimized_spend']:.2f}",
            change_str,
        )

    console.print(table)
    console.print(f"Expected ROI: {expected_roi:.2f}%", style="bold green")


def _get_specific_model_config(model_type: str, confidence: float) -> Dict[str, Dict]:
    """Get configuration for a specific model type."""
    if model_type == "baseline":
        return {
            "baseline_lr": {
                "type": "baseline",
                "params": {
                    "features": ["spend", "impressions", "clicks"],
                    "scale_features": True,
                    "confidence_level": confidence,
                },
            }
        }
    elif model_type == "arima":
        return {
            "arima": {
                "type": "time_series",
                "params": {"method": "arima", "confidence_level": confidence},
            }
        }
    elif model_type == "seasonal_naive":
        return {
            "seasonal_naive": {
                "type": "time_series",
                "params": {
                    "method": "seasonal_naive",
                    "seasonal_period": 7,
                    "confidence_level": confidence,
                },
            }
        }
    else:
        return None


def _create_synthetic_campaign_data(
    budget: float, start_date: Optional[datetime], end_date: Optional[datetime]
) -> pd.DataFrame:
    """Create synthetic campaign data for optimization demo."""
    if not start_date:
        start_date = datetime.now() - timedelta(days=30)
    if not end_date:
        end_date = datetime.now()

    # Create synthetic campaigns
    campaigns = [
        {
            "campaign_id": "campaign_001",
            "platform": "google",
            "campaign_type": "search",
        },
        {
            "campaign_id": "campaign_002",
            "platform": "facebook",
            "campaign_type": "social",
        },
        {
            "campaign_id": "campaign_003",
            "platform": "instagram",
            "campaign_type": "social",
        },
        {
            "campaign_id": "campaign_004",
            "platform": "google",
            "campaign_type": "display",
        },
        {
            "campaign_id": "campaign_005",
            "platform": "linkedin",
            "campaign_type": "social",
        },
    ]

    # Generate synthetic data
    data = []
    for campaign in campaigns:
        # Simulate varying performance
        base_spend = budget / len(campaigns) * np.random.uniform(0.5, 1.5)
        base_roi = np.random.uniform(0.8, 2.5)

        data.append(
            {
                "campaign_id": campaign["campaign_id"],
                "campaign_name": f"Campaign {campaign['campaign_id'].split('_')[1]}",
                "platform": campaign["platform"],
                "campaign_type": campaign["campaign_type"],
                "date": start_date,
                "spend": base_spend,
                "revenue": base_spend * base_roi,
                "impressions": int(base_spend * np.random.uniform(50, 200)),
                "clicks": int(base_spend * np.random.uniform(2, 10)),
                "conversions": int(base_spend * np.random.uniform(0.1, 1.0)),
                "target_audience": "general",
                "geographic_region": "US",
                "budget_daily": base_spend,
                "budget_total": base_spend * 30,
            }
        )

    return pd.DataFrame(data)


def _df_to_campaign_dataset(df: pd.DataFrame) -> CampaignDataset:
    """Convert DataFrame to CampaignDataset for optimization."""
    records = []
    for _, row in df.iterrows():
        record = CampaignRecord(
            campaign_id=row["campaign_id"],
            campaign_name=row.get("campaign_name", ""),
            campaign_type=row.get("campaign_type", ""),
            platform=row.get("platform", ""),
            date=row["date"],
            spend=float(row["spend"]),
            impressions=int(row["impressions"]),
            clicks=int(row["clicks"]),
            conversions=int(row["conversions"]),
            revenue=float(row["revenue"]),
            target_audience=row.get("target_audience", ""),
            geographic_region=row.get("geographic_region", ""),
            budget_daily=float(row.get("budget_daily", 0)),
            budget_total=float(row.get("budget_total", 0)),
        )
        records.append(record)

    return CampaignDataset(records=records)


def _create_optimization_results(
    df: pd.DataFrame, optimal_allocation: np.ndarray, expected_roi: float, budget: float
) -> List[Dict]:
    """Create optimization results from allocation."""
    # Group by campaign to get current spend
    campaign_spend = df.groupby("campaign_id")["spend"].sum().reset_index()

    results = []
    for i, (_, row) in enumerate(campaign_spend.iterrows()):
        if i < len(optimal_allocation):
            results.append(
                {
                    "campaign_id": row["campaign_id"],
                    "current_spend": row["spend"],
                    "optimized_spend": optimal_allocation[i],
                    "expected_roi": expected_roi,
                }
            )

    return results


if __name__ == "__main__":
    cli()
