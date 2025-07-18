"""
Visualization module for digital ad campaign ROI forecasting.

This module provides matplotlib/seaborn charts for visualizing campaign data,
forecasting results, and optimization outcomes.

Main functions:
- plot_spend_vs_conversions: Scatter plot with regression line
- plot_forecasted_conversions: Time series forecasting visualization
- plot_budget_allocation: Bar chart of optimized budget allocation & ROI
- plot_campaign_performance: Comprehensive performance dashboard
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import warnings

# Configure matplotlib and seaborn
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")
warnings.filterwarnings("ignore", category=FutureWarning)

# Set default figure parameters
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10


def plot_spend_vs_conversions(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    title: str = "Campaign Spend vs Conversions",
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Create a scatter plot of spend vs conversions with regression line.

    Args:
        df: DataFrame with 'spend' and 'conversions' columns
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
        title: Plot title
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create scatter plot with regression line
    sns.scatterplot(
        data=df,
        x="spend",
        y="conversions",
        hue="platform" if "platform" in df.columns else None,
        size="revenue" if "revenue" in df.columns else None,
        sizes=(50, 300),
        alpha=0.7,
        ax=ax,
    )

    # Add regression line
    sns.regplot(
        data=df,
        x="spend",
        y="conversions",
        scatter=False,
        color="red",
        line_kws={"linewidth": 2, "alpha": 0.8},
        ax=ax,
    )

    # Calculate and display correlation
    correlation = df["spend"].corr(df["conversions"])

    # Customize plot
    ax.set_xlabel("Campaign Spend ($)")
    ax.set_ylabel("Conversions")
    ax.set_title(f"{title}\nCorrelation: {correlation:.3f}")
    ax.grid(True, alpha=0.3)

    # Add trend line equation
    z = np.polyfit(df["spend"], df["conversions"], 1)
    p = np.poly1d(z)
    ax.text(
        0.05,
        0.95,
        f"Trend: y = {z[0]:.4f}x + {z[1]:.2f}",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        verticalalignment="top",
    )

    # Format axes
    ax.ticklabel_format(style="plain", axis="both")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()

    return fig


def plot_forecasted_conversions(
    historical_data: pd.DataFrame,
    forecast_data: pd.DataFrame,
    confidence_intervals: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    title: str = "Forecasted Conversions Over Time",
    figsize: Tuple[int, int] = (14, 8),
) -> plt.Figure:
    """
    Create a time series plot showing historical and forecasted conversions.

    Args:
        historical_data: DataFrame with 'date' and 'conversions' columns
        forecast_data: DataFrame with 'date' and 'conversions' columns for forecasts
        confidence_intervals: Optional DataFrame with confidence intervals
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
        title: Plot title
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Ensure date columns are datetime
    historical_data = historical_data.copy()
    forecast_data = forecast_data.copy()
    historical_data["date"] = pd.to_datetime(historical_data["date"])
    forecast_data["date"] = pd.to_datetime(forecast_data["date"])

    # Plot historical data
    ax.plot(
        historical_data["date"],
        historical_data["conversions"],
        label="Historical Data",
        color="blue",
        linewidth=2,
        marker="o",
        markersize=4,
    )

    # Plot forecast data
    ax.plot(
        forecast_data["date"],
        forecast_data["conversions"],
        label="Forecast",
        color="red",
        linewidth=2,
        linestyle="--",
        marker="s",
        markersize=4,
    )

    # Plot confidence intervals if provided
    if confidence_intervals is not None:
        confidence_intervals = confidence_intervals.copy()
        confidence_intervals["date"] = pd.to_datetime(confidence_intervals["date"])

        ax.fill_between(
            confidence_intervals["date"],
            confidence_intervals["lower_bound"],
            confidence_intervals["upper_bound"],
            alpha=0.2,
            color="red",
            label="Confidence Interval",
        )

    # Add vertical line to separate historical and forecast
    if len(historical_data) > 0 and len(forecast_data) > 0:
        cutoff_date = historical_data["date"].max()
        ax.axvline(x=cutoff_date, color="gray", linestyle=":", alpha=0.7)
        ax.text(
            cutoff_date,
            ax.get_ylim()[1] * 0.95,
            "Forecast Start",
            rotation=90,
            ha="right",
            va="top",
        )

    # Customize plot
    ax.set_xlabel("Date")
    ax.set_ylabel("Conversions")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.tick_params(axis="x", rotation=45)

    # Add summary statistics
    if len(forecast_data) > 0:
        avg_forecast = forecast_data["conversions"].mean()
        total_forecast = forecast_data["conversions"].sum()

        ax.text(
            0.02,
            0.98,
            f"Avg Forecast: {avg_forecast:.0f}\nTotal Forecast: {total_forecast:.0f}",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            verticalalignment="top",
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()

    return fig


def plot_budget_allocation(
    campaign_names: List[str],
    current_allocation: List[float],
    optimized_allocation: List[float],
    roi_values: List[float],
    save_path: Optional[str] = None,
    show_plot: bool = True,
    title: str = "Optimized Budget Allocation & ROI",
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Create a bar chart showing current vs optimized budget allocation and ROI.

    Args:
        campaign_names: List of campaign names
        current_allocation: List of current budget allocations
        optimized_allocation: List of optimized budget allocations
        roi_values: List of ROI values for each campaign
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
        title: Plot title
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # Prepare data
    x = np.arange(len(campaign_names))
    width = 0.35

    # Plot 1: Budget Allocation Comparison
    bars1 = ax1.bar(
        x - width / 2, current_allocation, width, label="Current Allocation", alpha=0.8
    )
    bars2 = ax1.bar(
        x + width / 2,
        optimized_allocation,
        width,
        label="Optimized Allocation",
        alpha=0.8,
    )

    # Add value labels on bars
    def add_value_labels(ax, bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"${height:,.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    add_value_labels(ax1, bars1)
    add_value_labels(ax1, bars2)

    ax1.set_xlabel("Campaigns")
    ax1.set_ylabel("Budget Allocation ($)")
    ax1.set_title("Budget Allocation: Current vs Optimized")
    ax1.set_xticks(x)
    ax1.set_xticklabels(campaign_names, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: ROI by Campaign
    bars3 = ax2.bar(x, roi_values, color="green", alpha=0.7)

    # Add value labels on ROI bars
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax2.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax2.set_xlabel("Campaigns")
    ax2.set_ylabel("ROI (%)")
    ax2.set_title("Expected ROI by Campaign")
    ax2.set_xticks(x)
    ax2.set_xticklabels(campaign_names, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3)

    # Add horizontal line at 0% ROI
    ax2.axhline(y=0, color="red", linestyle="--", alpha=0.5)

    # Add summary statistics
    total_current = sum(current_allocation)
    total_optimized = sum(optimized_allocation)
    avg_roi = sum(roi_values) / len(roi_values)

    fig.suptitle(
        f"{title}\nTotal Budget: ${total_current:,.0f} → ${total_optimized:,.0f} | Avg ROI: {avg_roi:.1f}%",
        fontsize=14,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()

    return fig


def plot_campaign_performance(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    title: str = "Campaign Performance Dashboard",
    figsize: Tuple[int, int] = (16, 12),
) -> plt.Figure:
    """
    Create a comprehensive performance dashboard with multiple subplots.

    Args:
        df: DataFrame with campaign data
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
        title: Plot title
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

    # Calculate derived metrics
    df = df.copy()
    df["roi"] = (df["revenue"] - df["spend"]) / df["spend"] * 100
    df["ctr"] = df["clicks"] / df["impressions"] * 100
    df["conversion_rate"] = df["conversions"] / df["clicks"] * 100
    df["cpc"] = df["spend"] / df["clicks"]

    # Plot 1: ROI by Platform
    if "platform" in df.columns:
        platform_roi = df.groupby("platform")["roi"].mean().sort_values(ascending=False)
        colors = sns.color_palette("husl", len(platform_roi))
        bars = ax1.bar(platform_roi.index, platform_roi.values, color=colors, alpha=0.8)

        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

        ax1.set_title("Average ROI by Platform")
        ax1.set_ylabel("ROI (%)")
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(True, alpha=0.3)

    # Plot 2: Spend vs Revenue Scatter
    if "platform" in df.columns:
        sns.scatterplot(
            data=df,
            x="spend",
            y="revenue",
            hue="platform",
            size="conversions",
            sizes=(50, 300),
            alpha=0.7,
            ax=ax2,
        )
    else:
        ax2.scatter(df["spend"], df["revenue"], alpha=0.7)

    # Add diagonal line for break-even
    max_val = max(df["spend"].max(), df["revenue"].max())
    ax2.plot([0, max_val], [0, max_val], "r--", alpha=0.5, label="Break-even")

    ax2.set_xlabel("Spend ($)")
    ax2.set_ylabel("Revenue ($)")
    ax2.set_title("Spend vs Revenue")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Conversion Rate by Campaign Type
    if "campaign_type" in df.columns:
        type_conversion = (
            df.groupby("campaign_type")["conversion_rate"]
            .mean()
            .sort_values(ascending=True)
        )
        ax3.barh(
            type_conversion.index, type_conversion.values, color="lightblue", alpha=0.8
        )

        for i, v in enumerate(type_conversion.values):
            ax3.text(v + 0.1, i, f"{v:.2f}%", va="center")

        ax3.set_xlabel("Conversion Rate (%)")
        ax3.set_title("Average Conversion Rate by Campaign Type")
        ax3.grid(True, alpha=0.3)

    # Plot 4: Time Series of Daily Performance
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        daily_performance = (
            df.groupby("date")
            .agg({"spend": "sum", "revenue": "sum", "conversions": "sum"})
            .reset_index()
        )

        ax4_twin = ax4.twinx()

        # Plot spend and revenue
        ax4.plot(
            daily_performance["date"],
            daily_performance["spend"],
            label="Spend",
            color="red",
            linewidth=2,
        )
        ax4.plot(
            daily_performance["date"],
            daily_performance["revenue"],
            label="Revenue",
            color="green",
            linewidth=2,
        )

        # Plot conversions on secondary axis
        ax4_twin.bar(
            daily_performance["date"],
            daily_performance["conversions"],
            alpha=0.3,
            color="blue",
            label="Conversions",
        )

        ax4.set_xlabel("Date")
        ax4.set_ylabel("Amount ($)")
        ax4_twin.set_ylabel("Conversions")
        ax4.set_title("Daily Performance Trends")
        ax4.tick_params(axis="x", rotation=45)
        ax4.legend(loc="upper left")
        ax4_twin.legend(loc="upper right")
        ax4.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()

    return fig


def create_roi_trend_plot(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    title: str = "ROI Trend Analysis",
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Create a ROI trend analysis plot over time.

    Args:
        df: DataFrame with campaign data including 'date' column
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
        title: Plot title
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["roi"] = (df["revenue"] - df["spend"]) / df["spend"] * 100

    # Calculate daily ROI
    daily_roi = (
        df.groupby("date")
        .apply(
            lambda x: (x["revenue"].sum() - x["spend"].sum()) / x["spend"].sum() * 100
        )
        .reset_index()
    )
    daily_roi.columns = ["date", "roi"]

    # Plot ROI trend
    ax.plot(daily_roi["date"], daily_roi["roi"], linewidth=2, marker="o", markersize=4)

    # Add trend line
    x_numeric = pd.to_numeric(daily_roi["date"])
    z = np.polyfit(x_numeric, daily_roi["roi"], 1)
    p = np.poly1d(z)
    ax.plot(
        daily_roi["date"],
        p(x_numeric),
        "r--",
        alpha=0.8,
        linewidth=2,
        label=f"Trend (slope: {z[0]:.4f})",
    )

    # Add horizontal line at 0% ROI
    ax.axhline(y=0, color="red", linestyle=":", alpha=0.5, label="Break-even")

    # Customize plot
    ax.set_xlabel("Date")
    ax.set_ylabel("ROI (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45)

    # Add summary statistics
    avg_roi = daily_roi["roi"].mean()
    min_roi = daily_roi["roi"].min()
    max_roi = daily_roi["roi"].max()

    ax.text(
        0.02,
        0.98,
        f"Avg ROI: {avg_roi:.1f}%\nMin ROI: {min_roi:.1f}%\nMax ROI: {max_roi:.1f}%",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        verticalalignment="top",
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()

    return fig


def save_all_plots(
    df: pd.DataFrame,
    output_dir: str,
    forecast_data: Optional[pd.DataFrame] = None,
    budget_allocation_data: Optional[Dict] = None,
    prefix: str = "campaign_analysis",
) -> List[str]:
    """
    Generate and save all available plots for the campaign data.

    Args:
        df: Main campaign DataFrame
        output_dir: Directory to save plots
        forecast_data: Optional forecast data
        budget_allocation_data: Optional budget allocation data
        prefix: Prefix for saved files

    Returns:
        List of saved file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = []

    # Plot 1: Spend vs Conversions
    file_path = output_path / f"{prefix}_spend_vs_conversions.png"
    plot_spend_vs_conversions(df, save_path=str(file_path), show_plot=False)
    saved_files.append(str(file_path))

    # Plot 2: Campaign Performance Dashboard
    file_path = output_path / f"{prefix}_performance_dashboard.png"
    plot_campaign_performance(df, save_path=str(file_path), show_plot=False)
    saved_files.append(str(file_path))

    # Plot 3: ROI Trend
    if "date" in df.columns:
        file_path = output_path / f"{prefix}_roi_trend.png"
        create_roi_trend_plot(df, save_path=str(file_path), show_plot=False)
        saved_files.append(str(file_path))

    # Plot 4: Forecasted Conversions (if forecast data available)
    if forecast_data is not None:
        file_path = output_path / f"{prefix}_forecasted_conversions.png"
        plot_forecasted_conversions(
            df, forecast_data, save_path=str(file_path), show_plot=False
        )
        saved_files.append(str(file_path))

    # Plot 5: Budget Allocation (if budget data available)
    if budget_allocation_data is not None:
        file_path = output_path / f"{prefix}_budget_allocation.png"
        plot_budget_allocation(
            budget_allocation_data["campaign_names"],
            budget_allocation_data["current_allocation"],
            budget_allocation_data["optimized_allocation"],
            budget_allocation_data["roi_values"],
            save_path=str(file_path),
            show_plot=False,
        )
        saved_files.append(str(file_path))

    return saved_files
