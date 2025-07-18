"""
Model selection module for forecasting models.

This module provides functionality to compare different forecasting models
using cross-validation and select the best performing model based on
various evaluation metrics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

from .baseline import BaselineForecaster, create_baseline_forecaster
from .time_series import TimeSeriesForecaster, create_time_series_forecaster

logger = logging.getLogger(__name__)


class ModelSelector:
    """
    Model selection and comparison class for forecasting models.

    This class provides functionality to compare different forecasting models
    using time series cross-validation and select the best performing model.
    """

    def __init__(
        self,
        models: Optional[Dict[str, Any]] = None,
        cv_folds: int = 5,
        scoring: str = "mae",
        confidence_level: float = 0.95,
    ):
        """
        Initialize the model selector.

        Args:
            models: Dictionary of models to compare. If None, uses default models.
            cv_folds: Number of cross-validation folds.
            scoring: Scoring metric for model selection ('mae', 'mse', 'rmse', 'mape').
            confidence_level: Confidence level for predictions.
        """
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.confidence_level = confidence_level

        # Default models if none provided
        if models is None:
            self.models = self._get_default_models()
        else:
            self.models = models

        # Results storage
        self.cv_results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = None
        self.is_fitted = False

    def _get_default_models(self) -> Dict[str, Any]:
        """
        Get default models for comparison.

        Returns:
            Dictionary of default models.
        """
        models = {
            "baseline_lr": {
                "type": "baseline",
                "params": {
                    "features": ["spend", "impressions", "clicks"],
                    "scale_features": True,
                    "confidence_level": self.confidence_level,
                },
            },
            "baseline_simple": {
                "type": "baseline",
                "params": {
                    "features": ["spend"],
                    "scale_features": False,
                    "confidence_level": self.confidence_level,
                },
            },
            "seasonal_naive": {
                "type": "time_series",
                "params": {
                    "method": "seasonal_naive",
                    "seasonal_period": 7,
                    "confidence_level": self.confidence_level,
                },
            },
        }

        # Add ARIMA model if statsmodels is available
        try:
            from .time_series import STATSMODELS_AVAILABLE

            if STATSMODELS_AVAILABLE:
                models["arima"] = {
                    "type": "time_series",
                    "params": {
                        "method": "arima",
                        "confidence_level": self.confidence_level,
                    },
                }
        except ImportError:
            pass

        return models

    def _create_model(self, model_name: str, model_config: Dict[str, Any]) -> Any:
        """
        Create a model instance from configuration.

        Args:
            model_name: Name of the model.
            model_config: Model configuration.

        Returns:
            Model instance.
        """
        model_type = model_config["type"]
        params = model_config["params"]

        if model_type == "baseline":
            return BaselineForecaster(**params)
        elif model_type == "time_series":
            return TimeSeriesForecaster(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _evaluate_model(
        self, model: Any, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Evaluate a model on train/test data.

        Args:
            model: Model instance.
            train_df: Training data.
            test_df: Testing data.

        Returns:
            Dictionary with evaluation metrics.
        """
        try:
            # Fit the model
            model.fit(train_df)

            # Make predictions
            if isinstance(model, BaselineForecaster):
                # For baseline models
                predictions = model.predict(test_df)
                y_true = test_df["conversions"].values
                y_pred = predictions["predicted_conversions"].values
            elif isinstance(model, TimeSeriesForecaster):
                # For time series models
                # Prepare test data for time series
                test_df_copy = test_df.copy()
                test_df_copy["date"] = pd.to_datetime(test_df_copy["date"])
                daily_conversions = (
                    test_df_copy.groupby("date")["conversions"].sum().reset_index()
                )
                daily_conversions = daily_conversions.sort_values("date")

                # Make predictions for the number of unique dates
                n_periods = len(daily_conversions)
                predictions = model.predict(periods=n_periods)

                y_true = daily_conversions["conversions"].values
                y_pred = predictions["predicted_conversions"].values[: len(y_true)]
            else:
                # Unknown model type
                raise ValueError(f"Unknown model type: {type(model)}")

            # Calculate metrics
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100

            # Calculate R² score (handle potential negative values)
            try:
                r2 = r2_score(y_true, y_pred)
            except:
                r2 = 0.0

            return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "r2_score": r2}

        except Exception as e:
            logger.warning(f"Error evaluating model: {e}")
            return {
                "mae": float("inf"),
                "mse": float("inf"),
                "rmse": float("inf"),
                "mape": float("inf"),
                "r2_score": -float("inf"),
            }

    def _time_series_cv_split(
        self, df: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create time series cross-validation splits.

        Args:
            df: Input dataframe.

        Returns:
            List of (train, test) dataframe tuples.
        """
        # Sort by date
        df_sorted = df.copy()
        df_sorted["date"] = pd.to_datetime(df_sorted["date"])
        df_sorted = df_sorted.sort_values("date")

        # Create time series splits
        n_samples = len(df_sorted)
        min_train_size = max(
            2, n_samples // (self.cv_folds + 1)
        )  # Reduced minimum train size

        splits = []
        for i in range(self.cv_folds):
            # Calculate split points
            train_end = (
                min_train_size + (i + 1) * (n_samples - min_train_size) // self.cv_folds
            )
            test_start = train_end
            test_end = min(
                test_start + max(1, (n_samples - min_train_size) // self.cv_folds),
                n_samples,
            )

            if (
                test_end > test_start and train_end >= 2
            ):  # Ensure we have at least 2 samples for training
                train_df = df_sorted.iloc[:train_end]
                test_df = df_sorted.iloc[test_start:test_end]
                splits.append((train_df, test_df))

        return splits

    def fit(self, df: pd.DataFrame) -> "ModelSelector":
        """
        Fit and evaluate all models using cross-validation.

        Args:
            df: Training dataframe.

        Returns:
            Self for method chaining.
        """
        logger.info("Starting model selection with cross-validation...")

        # Create time series splits
        cv_splits = self._time_series_cv_split(df)

        if not cv_splits:
            raise ValueError("Not enough data for cross-validation")

        # Evaluate each model
        for model_name, model_config in self.models.items():
            logger.info(f"Evaluating model: {model_name}")

            fold_scores = []

            for fold, (train_df, test_df) in enumerate(cv_splits):
                try:
                    # Create model instance
                    model = self._create_model(model_name, model_config)

                    # Evaluate model
                    scores = self._evaluate_model(model, train_df, test_df)
                    fold_scores.append(scores)

                    logger.debug(
                        f"Fold {fold + 1} - {model_name}: {scores[self.scoring]:.4f}"
                    )

                except Exception as e:
                    logger.warning(f"Error in fold {fold + 1} for {model_name}: {e}")
                    continue

            if fold_scores:
                # Calculate mean and std of scores
                mean_scores = {}
                std_scores = {}

                for metric in ["mae", "mse", "rmse", "mape", "r2_score"]:
                    scores = [
                        score[metric]
                        for score in fold_scores
                        if not np.isinf(score[metric])
                    ]
                    if scores:
                        mean_scores[metric] = np.mean(scores)
                        std_scores[metric] = np.std(scores)
                    else:
                        mean_scores[metric] = (
                            float("inf") if metric != "r2_score" else -float("inf")
                        )
                        std_scores[metric] = 0.0

                self.cv_results[model_name] = {
                    "mean_scores": mean_scores,
                    "std_scores": std_scores,
                    "fold_scores": fold_scores,
                    "n_folds": len(fold_scores),
                }

                logger.info(
                    f"Model {model_name} - Mean {self.scoring}: {mean_scores[self.scoring]:.4f} "
                    f"(±{std_scores[self.scoring]:.4f})"
                )
            else:
                logger.warning(f"No valid scores for model {model_name}")

        # Select best model
        self._select_best_model()

        # Fit best model on full data
        if self.best_model_name:
            self.best_model = self._create_model(
                self.best_model_name, self.models[self.best_model_name]
            )
            self.best_model.fit(df)
            self.is_fitted = True

            logger.info(
                f"Best model selected: {self.best_model_name} "
                f"(score: {self.best_score:.4f})"
            )

        return self

    def _select_best_model(self):
        """Select the best model based on cross-validation results."""
        if not self.cv_results:
            return

        best_score = float("inf") if self.scoring != "r2_score" else -float("inf")
        best_model_name = None

        for model_name, results in self.cv_results.items():
            score = results["mean_scores"][self.scoring]

            if self.scoring == "r2_score":
                # For R², higher is better
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
            else:
                # For error metrics, lower is better
                if score < best_score:
                    best_score = score
                    best_model_name = model_name

        self.best_model_name = best_model_name
        self.best_score = best_score

    def predict(self, df: pd.DataFrame = None, periods: int = 30) -> pd.DataFrame:
        """
        Make predictions using the best model.

        Args:
            df: Input dataframe for baseline models (optional).
            periods: Number of periods to forecast for time series models.

        Returns:
            DataFrame with predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model selector must be fitted before making predictions")

        if self.best_model is None:
            raise ValueError("No best model found")

        # Make predictions based on model type
        if isinstance(self.best_model, BaselineForecaster):
            if df is None:
                raise ValueError("DataFrame required for baseline model predictions")
            return self.best_model.predict(df)
        elif isinstance(self.best_model, TimeSeriesForecaster):
            return self.best_model.predict(periods)
        else:
            raise ValueError(f"Unknown model type: {type(self.best_model)}")

    def get_cv_results(self) -> Dict[str, Dict[str, Any]]:
        """
        Get cross-validation results.

        Returns:
            Dictionary with cross-validation results for each model.
        """
        return self.cv_results

    def get_best_model_info(self) -> Dict[str, Any]:
        """
        Get information about the best model.

        Returns:
            Dictionary with best model information.
        """
        if not self.is_fitted:
            return {}

        return {
            "name": self.best_model_name,
            "score": self.best_score,
            "scoring_metric": self.scoring,
            "cv_results": self.cv_results.get(self.best_model_name, {}),
            "model_type": (
                self.models[self.best_model_name]["type"]
                if self.best_model_name
                else None
            ),
        }

    def get_model_comparison(self) -> pd.DataFrame:
        """
        Get a comparison of all models.

        Returns:
            DataFrame with model comparison results.
        """
        if not self.cv_results:
            return pd.DataFrame()

        comparison_data = []

        for model_name, results in self.cv_results.items():
            mean_scores = results["mean_scores"]
            std_scores = results["std_scores"]

            row = {
                "model": model_name,
                "model_type": self.models[model_name]["type"],
                "n_folds": results["n_folds"],
            }

            # Add mean scores
            for metric in ["mae", "mse", "rmse", "mape", "r2_score"]:
                row[f"{metric}_mean"] = mean_scores[metric]
                row[f"{metric}_std"] = std_scores[metric]

            # Add ranking
            row["is_best"] = model_name == self.best_model_name

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Sort by the scoring metric
        if self.scoring == "r2_score":
            df = df.sort_values(f"{self.scoring}_mean", ascending=False)
        else:
            df = df.sort_values(f"{self.scoring}_mean", ascending=True)

        return df


def select_best_model(
    df: pd.DataFrame,
    models: Optional[Dict[str, Any]] = None,
    cv_folds: int = 5,
    scoring: str = "mae",
    confidence_level: float = 0.95,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Convenience function to select the best forecasting model.

    Args:
        df: Training dataframe.
        models: Dictionary of models to compare.
        cv_folds: Number of cross-validation folds.
        scoring: Scoring metric for model selection.
        confidence_level: Confidence level for predictions.

    Returns:
        Tuple of (best_model, model_info).
    """
    selector = ModelSelector(
        models=models,
        cv_folds=cv_folds,
        scoring=scoring,
        confidence_level=confidence_level,
    )

    selector.fit(df)

    return selector.best_model, selector.get_best_model_info()


def compare_models(
    df: pd.DataFrame,
    models: Optional[Dict[str, Any]] = None,
    cv_folds: int = 5,
    scoring: str = "mae",
) -> pd.DataFrame:
    """
    Compare multiple forecasting models using cross-validation.

    Args:
        df: Training dataframe.
        models: Dictionary of models to compare.
        cv_folds: Number of cross-validation folds.
        scoring: Scoring metric for comparison.

    Returns:
        DataFrame with model comparison results.
    """
    selector = ModelSelector(models=models, cv_folds=cv_folds, scoring=scoring)

    selector.fit(df)

    return selector.get_model_comparison()


def forecast_with_best_model(
    df: pd.DataFrame,
    future_periods: int = 30,
    spend_scenario: Optional[Union[float, List[float]]] = None,
    models: Optional[Dict[str, Any]] = None,
    cv_folds: int = 5,
    scoring: str = "mae",
    confidence_level: float = 0.95,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Select the best model and make forecasts.

    Args:
        df: Historical campaign data.
        future_periods: Number of future periods to forecast.
        spend_scenario: Spend scenario for baseline models.
        models: Dictionary of models to compare.
        cv_folds: Number of cross-validation folds.
        scoring: Scoring metric for model selection.
        confidence_level: Confidence level for predictions.

    Returns:
        Tuple of (forecasts, model_info).
    """
    # Select best model
    selector = ModelSelector(
        models=models,
        cv_folds=cv_folds,
        scoring=scoring,
        confidence_level=confidence_level,
    )

    selector.fit(df)

    # Make forecasts
    if isinstance(selector.best_model, BaselineForecaster):
        # For baseline models, use spend scenario
        if spend_scenario is None:
            spend_scenario = df["spend"].mean()

        forecasts = selector.best_model.forecast_future(
            df, future_periods, spend_scenario
        )
    elif isinstance(selector.best_model, TimeSeriesForecaster):
        # For time series models, predict directly
        forecasts = selector.best_model.predict(future_periods)
    else:
        raise ValueError(f"Unknown model type: {type(selector.best_model)}")

    # Add model information
    forecasts["model_used"] = selector.best_model_name

    return forecasts, selector.get_best_model_info()


class EnsembleForecaster:
    """
    Ensemble forecasting model that combines multiple models.

    This class creates weighted averages of predictions from multiple models
    to potentially improve forecast accuracy.
    """

    def __init__(
        self,
        models: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
        confidence_level: float = 0.95,
    ):
        """
        Initialize the ensemble forecaster.

        Args:
            models: Dictionary of models to ensemble.
            weights: Weights for each model. If None, uses equal weights.
            confidence_level: Confidence level for predictions.
        """
        self.models = models
        self.weights = weights
        self.confidence_level = confidence_level
        self.fitted_models = {}
        self.is_fitted = False

        # Set equal weights if not provided
        if self.weights is None:
            n_models = len(models)
            self.weights = {name: 1.0 / n_models for name in models.keys()}

        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {
            name: weight / total_weight for name, weight in self.weights.items()
        }

    def fit(self, df: pd.DataFrame) -> "EnsembleForecaster":
        """
        Fit all models in the ensemble.

        Args:
            df: Training dataframe.

        Returns:
            Self for method chaining.
        """
        logger.info("Training ensemble forecasting models...")

        selector = ModelSelector(models=self.models)

        for model_name, model_config in self.models.items():
            try:
                model = selector._create_model(model_name, model_config)
                model.fit(df)
                self.fitted_models[model_name] = model
                logger.info(f"Fitted model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to fit model {model_name}: {e}")

        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame = None, periods: int = 30) -> pd.DataFrame:
        """
        Make ensemble predictions.

        Args:
            df: Input dataframe for baseline models.
            periods: Number of periods to forecast.

        Returns:
            DataFrame with ensemble predictions.
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")

        if not self.fitted_models:
            raise ValueError("No successfully fitted models in ensemble")

        # Collect predictions from all models
        all_predictions = []
        model_weights = []

        for model_name, model in self.fitted_models.items():
            try:
                if isinstance(model, BaselineForecaster):
                    if df is None:
                        continue
                    pred = model.predict(df)
                elif isinstance(model, TimeSeriesForecaster):
                    pred = model.predict(periods)
                else:
                    continue

                all_predictions.append(pred["predicted_conversions"].values)
                model_weights.append(self.weights[model_name])

            except Exception as e:
                logger.warning(f"Error making prediction with {model_name}: {e}")
                continue

        if not all_predictions:
            raise ValueError("No successful predictions from ensemble models")

        # Calculate weighted average
        predictions_array = np.array(all_predictions)
        weights_array = np.array(model_weights)
        weights_array = weights_array / weights_array.sum()  # Renormalize

        ensemble_predictions = np.average(
            predictions_array, axis=0, weights=weights_array
        )

        # Calculate ensemble uncertainty (simplified)
        ensemble_std = np.std(predictions_array, axis=0)

        # Create results dataframe
        # Use the first successful prediction's structure as template
        template_pred = None
        for model_name, model in self.fitted_models.items():
            try:
                if isinstance(model, BaselineForecaster) and df is not None:
                    template_pred = model.predict(df)
                    break
                elif isinstance(model, TimeSeriesForecaster):
                    template_pred = model.predict(periods)
                    break
            except:
                continue

        if template_pred is None:
            raise ValueError("Could not create template prediction")

        # Create ensemble results
        results = template_pred.copy()
        results["predicted_conversions"] = ensemble_predictions
        results["ensemble_std"] = ensemble_std

        # Update confidence intervals
        from scipy import stats

        alpha = 1 - self.confidence_level
        t_val = stats.t.ppf(1 - alpha / 2, df=len(ensemble_predictions) - 1)
        margin_of_error = t_val * ensemble_std

        results["lower_bound"] = np.maximum(0, ensemble_predictions - margin_of_error)
        results["upper_bound"] = ensemble_predictions + margin_of_error
        results["model_used"] = "ensemble"

        return results
