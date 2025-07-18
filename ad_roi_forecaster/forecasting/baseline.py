"""
Baseline forecasting model using OLS/linear regression.

This module implements a simple baseline forecasting model that uses
ordinary least squares (OLS) linear regression to predict conversions
based on spend and other campaign features.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)


class BaselineForecaster:
    """
    Baseline forecasting model using linear regression.

    This model predicts conversions based on spend and other campaign features
    using simple linear regression. It provides a baseline for comparison
    with more sophisticated forecasting models.
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        scale_features: bool = True,
        confidence_level: float = 0.95,
    ):
        """
        Initialize the baseline forecaster.

        Args:
            features: List of feature columns to use. If None, uses default features.
            scale_features: Whether to scale features before training.
            confidence_level: Confidence level for prediction intervals.
        """
        self.features = features or ["spend", "impressions", "clicks"]
        self.scale_features = scale_features
        self.confidence_level = confidence_level

        # Model components
        self.model = LinearRegression()
        self.scaler = StandardScaler() if scale_features else None
        self.feature_names = None
        self.target_name = "conversions"

        # Training metadata
        self.is_trained = False
        self.train_score = None
        self.feature_importance = None
        self.residual_std = None

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for modeling.

        Args:
            df: Input dataframe with campaign data.

        Returns:
            DataFrame with prepared features.
        """
        # Start with basic features
        feature_df = df[self.features].copy()

        # Add derived features
        if "spend" in self.features and "impressions" in self.features:
            feature_df["cpm"] = np.where(
                df["impressions"] > 0, df["spend"] / df["impressions"] * 1000, 0
            )

        if "clicks" in self.features and "impressions" in self.features:
            feature_df["ctr"] = np.where(
                df["impressions"] > 0, df["clicks"] / df["impressions"], 0
            )

        if "spend" in self.features and "clicks" in self.features:
            feature_df["cpc"] = np.where(
                df["clicks"] > 0, df["spend"] / df["clicks"], 0
            )

        # Add temporal features if date column exists
        if "date" in df.columns:
            df_copy = df.copy()
            df_copy["date"] = pd.to_datetime(df_copy["date"])

            feature_df["day_of_week"] = df_copy["date"].dt.dayofweek
            feature_df["month"] = df_copy["date"].dt.month
            feature_df["quarter"] = df_copy["date"].dt.quarter

            # Add trend feature (days since start)
            min_date = df_copy["date"].min()
            feature_df["days_since_start"] = (df_copy["date"] - min_date).dt.days

        # Add categorical features if they exist
        categorical_features = ["platform", "campaign_type"]
        for cat_feature in categorical_features:
            if cat_feature in df.columns:
                # One-hot encode categorical features
                dummies = pd.get_dummies(df[cat_feature], prefix=cat_feature)
                feature_df = pd.concat([feature_df, dummies], axis=1)

        # Handle missing values
        feature_df = feature_df.fillna(0)

        # Store feature names for later use (only during training)
        if not self.is_trained:
            self.feature_names = feature_df.columns.tolist()
        elif self.feature_names is not None:
            # If we're already trained, ensure we have all the expected features
            for feature in self.feature_names:
                if feature not in feature_df.columns:
                    feature_df[feature] = 0

            # Reorder columns to match training order
            feature_df = feature_df[self.feature_names]

        return feature_df

    def fit(self, df: pd.DataFrame) -> "BaselineForecaster":
        """
        Fit the baseline model to the training data.

        Args:
            df: Training dataframe with campaign data.

        Returns:
            Self for method chaining.
        """
        logger.info("Training baseline forecasting model...")

        # Prepare features and target
        X = self._prepare_features(df)
        y = df[self.target_name].values

        # Validate that we have enough data
        if len(X) < 2:
            raise ValueError("Need at least 2 samples to train the model")

        # Scale features if requested
        if self.scale_features:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.values

        # Train the model
        self.model.fit(X_scaled, y)

        # Calculate training metrics
        y_pred = self.model.predict(X_scaled)
        self.train_score = r2_score(y, y_pred)

        # Calculate residual standard deviation for confidence intervals
        residuals = y - y_pred
        self.residual_std = np.std(residuals)

        # Store feature importance (absolute values of coefficients)
        if self.scale_features:
            # For scaled features, importance is just the absolute coefficients
            self.feature_importance = dict(
                zip(self.feature_names, np.abs(self.model.coef_))
            )
        else:
            # For unscaled features, normalize by feature std
            feature_stds = np.std(X, axis=0)
            normalized_coefs = np.abs(self.model.coef_) * feature_stds
            self.feature_importance = dict(zip(self.feature_names, normalized_coefs))

        self.is_trained = True
        logger.info(f"Model trained successfully. R² score: {self.train_score:.4f}")

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions using the trained model.

        Args:
            df: Dataframe with campaign data for prediction.

        Returns:
            DataFrame with predictions and confidence intervals.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Prepare features
        X = self._prepare_features(df)

        # Scale features if needed
        if self.scale_features:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values

        # Make predictions
        y_pred = self.model.predict(X_scaled)

        # Calculate confidence intervals
        # For linear regression, we use the residual standard deviation
        # and t-distribution for confidence intervals
        from scipy import stats

        alpha = 1 - self.confidence_level
        n = len(X)
        t_val = stats.t.ppf(1 - alpha / 2, df=n - 1)

        # Standard error of prediction (simplified)
        # In practice, this should account for the uncertainty in the model parameters
        prediction_std = self.residual_std
        margin_of_error = t_val * prediction_std

        # Create results dataframe
        results = pd.DataFrame(
            {
                "predicted_conversions": y_pred,
                "lower_bound": y_pred - margin_of_error,
                "upper_bound": y_pred + margin_of_error,
                "prediction_std": prediction_std,
                "confidence_level": self.confidence_level,
            }
        )

        # Ensure non-negative predictions
        results["predicted_conversions"] = np.maximum(
            0, results["predicted_conversions"]
        )
        results["lower_bound"] = np.maximum(0, results["lower_bound"])
        results["upper_bound"] = np.maximum(0, results["upper_bound"])

        # Add original data columns if they exist
        if "date" in df.columns:
            results["date"] = df["date"].values
        if "campaign_id" in df.columns:
            results["campaign_id"] = df["campaign_id"].values

        return results

    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            df: Test dataframe with campaign data.

        Returns:
            Dictionary with evaluation metrics.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        # Make predictions
        predictions = self.predict(df)
        y_true = df[self.target_name].values
        y_pred = predictions["predicted_conversions"].values

        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        # Calculate mean absolute percentage error (MAPE)
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100

        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2_score": r2,
            "mape": mape,
            "train_r2": self.train_score,
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")

        return self.feature_importance

    def forecast_future(
        self,
        df: pd.DataFrame,
        future_periods: int,
        spend_scenario: Union[float, List[float], Dict[str, float]],
    ) -> pd.DataFrame:
        """
        Forecast conversions for future periods.

        Args:
            df: Historical campaign data for context.
            future_periods: Number of future periods to forecast.
            spend_scenario: Spend scenario for forecasting. Can be:
                - float: constant spend per period
                - List[float]: spend values for each period
                - Dict[str, float]: spend values by campaign_id

        Returns:
            DataFrame with future predictions.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before forecasting")

        # Get the most recent date from the data
        df_copy = df.copy()
        df_copy["date"] = pd.to_datetime(df_copy["date"])
        latest_date = df_copy["date"].max()

        # Create future date range
        future_dates = pd.date_range(
            start=latest_date + timedelta(days=1), periods=future_periods, freq="D"
        )

        # Prepare future scenario data
        future_data = []

        for i, date in enumerate(future_dates):
            # Determine spend for this period
            if isinstance(spend_scenario, (int, float)):
                spend = spend_scenario
            elif isinstance(spend_scenario, list):
                spend = spend_scenario[i % len(spend_scenario)]
            else:
                # For dict scenario, we'll create one row per campaign
                for campaign_id, spend in spend_scenario.items():
                    row = self._create_future_row(df_copy, date, spend, campaign_id)
                    future_data.append(row)
                continue

            # Create future row with average values from historical data
            row = self._create_future_row(df_copy, date, spend)
            future_data.append(row)

        # Create future dataframe
        future_df = pd.DataFrame(future_data)

        # Make predictions
        predictions = self.predict(future_df)

        return predictions

    def _create_future_row(
        self,
        df: pd.DataFrame,
        date: pd.Timestamp,
        spend: float,
        campaign_id: Optional[str] = None,
    ) -> Dict:
        """
        Create a future row for forecasting.

        Args:
            df: Historical data for context.
            date: Future date.
            spend: Spend amount.
            campaign_id: Campaign ID if specified.

        Returns:
            Dictionary representing a future row.
        """
        # Calculate averages from historical data
        avg_data = df.mean(numeric_only=True)

        # Create base row
        row = {
            "date": date,
            "spend": spend,
        }

        # Add campaign_id if specified
        if campaign_id:
            row["campaign_id"] = campaign_id
            # Use campaign-specific averages if available
            campaign_data = df[df["campaign_id"] == campaign_id]
            if not campaign_data.empty:
                avg_data = campaign_data.mean(numeric_only=True)

        # Estimate other metrics based on historical relationships
        # These are simplified assumptions - in practice, you might want more sophisticated modeling

        # Assume impressions scale with spend
        if "impressions" in avg_data and avg_data["spend"] > 0:
            row["impressions"] = int(
                spend * avg_data["impressions"] / avg_data["spend"]
            )
        else:
            row["impressions"] = int(spend * 100)  # Default assumption

        # Assume clicks based on historical CTR
        if (
            "clicks" in avg_data
            and "impressions" in avg_data
            and avg_data["impressions"] > 0
        ):
            historical_ctr = avg_data["clicks"] / avg_data["impressions"]
            row["clicks"] = int(row["impressions"] * historical_ctr)
        else:
            row["clicks"] = int(row["impressions"] * 0.02)  # Default 2% CTR

        # Add categorical features (use most common values)
        categorical_features = ["platform", "campaign_type"]
        for cat_feature in categorical_features:
            if cat_feature in df.columns:
                if campaign_id and campaign_id in df["campaign_id"].values:
                    # Use campaign-specific value
                    campaign_data = df[df["campaign_id"] == campaign_id]
                    row[cat_feature] = campaign_data[cat_feature].iloc[0]
                else:
                    # Use most common value
                    row[cat_feature] = df[cat_feature].mode().iloc[0]

        return row


def create_baseline_forecaster(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[BaselineForecaster, Dict[str, float]]:
    """
    Create and train a baseline forecasting model.

    Args:
        df: Campaign data for training.
        test_size: Fraction of data to use for testing.
        random_state: Random state for reproducibility.

    Returns:
        Tuple of (trained model, evaluation metrics).
    """
    # Split data
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, shuffle=True
    )

    # Create and train model
    model = BaselineForecaster()
    model.fit(train_df)

    # Evaluate model
    metrics = model.evaluate(test_df)

    return model, metrics


def forecast_conversions_baseline(
    df: pd.DataFrame,
    future_periods: int = 30,
    spend_scenario: Union[float, List[float]] = None,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """
    Convenience function to create baseline forecasts.

    Args:
        df: Historical campaign data.
        future_periods: Number of future periods to forecast.
        spend_scenario: Spend scenario for forecasting.
        confidence_level: Confidence level for predictions.

    Returns:
        DataFrame with forecasted conversions and confidence intervals.
    """
    # Use average historical spend if no scenario provided
    if spend_scenario is None:
        spend_scenario = df["spend"].mean()

    # Create and train model
    model = BaselineForecaster(confidence_level=confidence_level)
    model.fit(df)

    # Make forecasts
    forecasts = model.forecast_future(df, future_periods, spend_scenario)

    return forecasts
