"""
Data loader module for reading and processing campaign data from CSV files.

This module provides functionality to load campaign data from CSV files,
perform data type coercion, and convert the data to pandas DataFrames.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import logging
from .schemas import CampaignRecord, CampaignDataset, ValidationResult

# Configure logging
logger = logging.getLogger(__name__)


class CampaignDataLoader:
    """
    Loader for campaign data from CSV files.

    Handles CSV reading, data type coercion, and conversion to structured formats.
    """

    # Default column mappings for common CSV formats
    DEFAULT_COLUMN_MAPPING = {
        "campaign_id": ["campaign_id", "Campaign_ID", "campaignId", "id"],
        "campaign_name": ["campaign_name", "Campaign_Name", "campaignName", "name"],
        "campaign_type": ["campaign_type", "Campaign_Type", "campaignType", "type"],
        "platform": ["platform", "Platform", "source", "Source"],
        "date": ["date", "Date", "day", "Day", "timestamp"],
        "spend": ["spend", "Spend", "cost", "Cost", "investment"],
        "impressions": ["impressions", "Impressions", "impr", "Impr"],
        "clicks": ["clicks", "Clicks", "click", "Click"],
        "conversions": ["conversions", "Conversions", "conv", "Conv"],
        "revenue": ["revenue", "Revenue", "sales", "Sales", "value"],
        "target_audience": [
            "target_audience",
            "Target_Audience",
            "audience",
            "segment",
        ],
        "geographic_region": [
            "geographic_region",
            "Geographic_Region",
            "region",
            "geo",
        ],
        "budget_daily": ["budget_daily", "Budget_Daily", "daily_budget", "dailyBudget"],
        "budget_total": ["budget_total", "Budget_Total", "total_budget", "totalBudget"],
    }

    # Data type specifications for each column
    DTYPE_SPECS = {
        "campaign_id": str,
        "campaign_name": str,
        "campaign_type": str,
        "platform": str,
        "date": "datetime64[ns]",
        "spend": float,
        "impressions": int,
        "clicks": int,
        "conversions": int,
        "revenue": float,
        "target_audience": str,
        "geographic_region": str,
        "budget_daily": float,
        "budget_total": float,
    }

    def __init__(self, column_mapping: Optional[Dict[str, List[str]]] = None):
        """
        Initialize the data loader.

        Args:
            column_mapping: Custom column mapping dictionary. If None, uses default.
        """
        self.column_mapping = column_mapping or self.DEFAULT_COLUMN_MAPPING
        self.logger = logging.getLogger(self.__class__.__name__)

    def _detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Detect column mappings in the DataFrame.

        Args:
            df: Input DataFrame with raw column names

        Returns:
            Dictionary mapping standard column names to actual column names
        """
        detected_mapping = {}
        available_columns = set(df.columns)

        for standard_name, possible_names in self.column_mapping.items():
            found_column = None
            for possible_name in possible_names:
                if possible_name in available_columns:
                    found_column = possible_name
                    break

            if found_column:
                detected_mapping[standard_name] = found_column
                self.logger.debug(f"Mapped {standard_name} -> {found_column}")
            else:
                self.logger.warning(f"Could not find column for {standard_name}")

        return detected_mapping

    def _coerce_data_types(
        self, df: pd.DataFrame, column_mapping: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Coerce data types according to schema specifications.

        Args:
            df: Input DataFrame
            column_mapping: Mapping of standard names to actual column names

        Returns:
            DataFrame with corrected data types
        """
        df_copy = df.copy()

        for standard_name, actual_name in column_mapping.items():
            if actual_name not in df_copy.columns:
                continue

            target_dtype = self.DTYPE_SPECS.get(standard_name)
            if target_dtype is None:
                continue

            try:
                if target_dtype == "datetime64[ns]":
                    # Handle datetime conversion with multiple formats
                    df_copy[actual_name] = pd.to_datetime(
                        df_copy[actual_name],
                        infer_datetime_format=True,
                        errors="coerce",
                    )
                elif target_dtype == float:
                    # Handle numeric conversion with error handling
                    df_copy[actual_name] = pd.to_numeric(
                        df_copy[actual_name], errors="coerce"
                    ).astype(float)
                elif target_dtype == int:
                    # Handle integer conversion with error handling
                    df_copy[actual_name] = (
                        pd.to_numeric(df_copy[actual_name], errors="coerce")
                        .fillna(0)
                        .astype(int)
                    )
                elif target_dtype == str:
                    # Handle string conversion
                    df_copy[actual_name] = df_copy[actual_name].astype(str)

                self.logger.debug(f"Converted {actual_name} to {target_dtype}")

            except Exception as e:
                self.logger.warning(
                    f"Failed to convert {actual_name} to {target_dtype}: {e}"
                )

        return df_copy

    def _standardize_column_names(
        self, df: pd.DataFrame, column_mapping: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Standardize column names according to schema.

        Args:
            df: Input DataFrame
            column_mapping: Mapping of standard names to actual column names

        Returns:
            DataFrame with standardized column names
        """
        rename_mapping = {
            actual: standard for standard, actual in column_mapping.items()
        }
        df_renamed = df.rename(columns=rename_mapping)

        # Ensure all required columns are present, fill missing with defaults
        required_columns = [
            "campaign_id",
            "campaign_name",
            "campaign_type",
            "platform",
            "date",
            "spend",
            "impressions",
            "clicks",
            "conversions",
            "revenue",
        ]

        for col in required_columns:
            if col not in df_renamed.columns:
                if col in ["spend", "revenue"]:
                    df_renamed[col] = 0.0
                elif col in ["impressions", "clicks", "conversions"]:
                    df_renamed[col] = 0
                else:
                    df_renamed[col] = ""

                self.logger.warning(
                    f"Missing required column {col}, filled with defaults"
                )

        return df_renamed

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the data.

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()

        # Remove rows with missing critical data
        initial_count = len(df_clean)
        df_clean = df_clean.dropna(subset=["campaign_id", "date"])

        if len(df_clean) < initial_count:
            self.logger.warning(
                f"Removed {initial_count - len(df_clean)} rows with missing critical data"
            )

        # Fill missing numeric values with 0
        numeric_columns = [
            "spend",
            "impressions",
            "clicks",
            "conversions",
            "revenue",
            "budget_daily",
            "budget_total",
        ]

        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(0)

        # Fill missing string values with empty string
        string_columns = [
            "campaign_name",
            "campaign_type",
            "platform",
            "target_audience",
            "geographic_region",
        ]

        for col in string_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna("")

        # Convert platform and campaign_type to lowercase for consistency
        if "platform" in df_clean.columns:
            df_clean["platform"] = df_clean["platform"].str.lower()

        if "campaign_type" in df_clean.columns:
            df_clean["campaign_type"] = df_clean["campaign_type"].str.lower()

        return df_clean

    def load_csv(
        self,
        file_path: Union[str, Path],
        encoding: str = "utf-8",
        delimiter: str = ",",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load campaign data from CSV file.

        Args:
            file_path: Path to the CSV file
            encoding: File encoding (default: utf-8)
            delimiter: CSV delimiter (default: comma)
            **kwargs: Additional arguments for pandas.read_csv

        Returns:
            pandas DataFrame with loaded and processed data

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file cannot be processed
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        self.logger.info(f"Loading CSV file: {file_path}")

        try:
            # Read CSV with flexible parsing
            df = pd.read_csv(
                file_path, encoding=encoding, delimiter=delimiter, **kwargs
            )

            if df.empty:
                raise ValueError("CSV file is empty")

            self.logger.info(f"Loaded {len(df)} rows from CSV")

            # Detect column mappings
            column_mapping = self._detect_columns(df)

            # Coerce data types
            df = self._coerce_data_types(df, column_mapping)

            # Standardize column names
            df = self._standardize_column_names(df, column_mapping)

            # Clean data
            df = self._clean_data(df)

            self.logger.info(f"Successfully processed {len(df)} rows")

            return df

        except Exception as e:
            self.logger.error(f"Error loading CSV file: {e}")
            raise ValueError(f"Failed to load CSV file: {e}")

    def load_multiple_csvs(
        self, file_paths: List[Union[str, Path]], **kwargs
    ) -> pd.DataFrame:
        """
        Load and combine multiple CSV files.

        Args:
            file_paths: List of paths to CSV files
            **kwargs: Additional arguments for load_csv

        Returns:
            Combined pandas DataFrame
        """
        dfs = []

        for file_path in file_paths:
            df = self.load_csv(file_path, **kwargs)
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)

        # Remove duplicates based on campaign_id and date
        combined_df = combined_df.drop_duplicates(subset=["campaign_id", "date"])

        self.logger.info(
            f"Combined {len(file_paths)} CSV files into {len(combined_df)} rows"
        )

        return combined_df


def load_campaign_data(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Main function to load campaign data from CSV file(s).

    This is the primary interface for loading campaign data. It handles
    both single files and directories containing multiple CSV files.

    Args:
        path: Path to CSV file or directory containing CSV files
        **kwargs: Additional arguments for CSV loading

    Returns:
        pandas DataFrame with loaded and processed campaign data

    Raises:
        FileNotFoundError: If the path doesn't exist
        ValueError: If no valid CSV files are found or data cannot be processed
    """
    path = Path(path)
    loader = CampaignDataLoader()

    if path.is_file():
        # Single file
        if path.suffix.lower() != ".csv":
            raise ValueError("File must be a CSV file")
        return loader.load_csv(path, **kwargs)

    elif path.is_dir():
        # Directory with multiple CSV files
        csv_files = list(path.glob("*.csv"))

        if not csv_files:
            raise ValueError(f"No CSV files found in directory: {path}")

        logger.info(f"Found {len(csv_files)} CSV files in directory")
        return loader.load_multiple_csvs(csv_files, **kwargs)

    else:
        raise FileNotFoundError(f"Path does not exist: {path}")
