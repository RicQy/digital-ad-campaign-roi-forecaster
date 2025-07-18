from pathlib import Path
from typing import List, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration settings for Digital Ad Campaign ROI Forecaster."""

    # ==================== PATHS ====================
    data_path: Path = Field(
        default=Path("./sample_data"), description="Path to sample data directory"
    )
    model_path: Path = Field(
        default=Path("./models"), description="Path to model storage directory"
    )
    output_path: Path = Field(
        default=Path("./outputs"), description="Path to output directory"
    )
    log_path: Path = Field(default=Path("./logs"), description="Path to log files")

    # ==================== MODEL HYPERPARAMETERS ====================
    # General ML parameters
    learning_rate: float = Field(
        default=0.001, ge=0.0001, le=1.0, description="Learning rate for training"
    )
    batch_size: int = Field(
        default=32, ge=1, le=1024, description="Batch size for training"
    )
    num_epochs: int = Field(
        default=100, ge=1, le=1000, description="Number of training epochs"
    )
    validation_split: float = Field(
        default=0.2, ge=0.1, le=0.5, description="Validation data split ratio"
    )

    # ROI forecasting specific parameters
    forecast_horizon: int = Field(
        default=30, ge=1, le=365, description="Forecasting horizon in days"
    )
    confidence_interval: float = Field(
        default=0.95, ge=0.8, le=0.99, description="Confidence interval for predictions"
    )
    seasonality_periods: List[int] = Field(
        default=[7, 30, 365], description="Seasonality periods to consider"
    )

    # Feature engineering
    lag_features: int = Field(
        default=7, ge=1, le=30, description="Number of lag features to create"
    )
    rolling_window_size: int = Field(
        default=7, ge=1, le=30, description="Rolling window size for features"
    )

    # ==================== OPTIMIZATION CONSTRAINTS ====================
    max_iterations: int = Field(
        default=1000, ge=10, le=10000, description="Maximum optimization iterations"
    )
    convergence_threshold: float = Field(
        default=1e-6, ge=1e-10, le=1e-3, description="Convergence threshold"
    )
    max_training_time: int = Field(
        default=3600, ge=60, le=86400, description="Maximum training time in seconds"
    )

    # Budget and ROI constraints
    min_roi_threshold: float = Field(
        default=1.0, ge=0.0, description="Minimum ROI threshold"
    )
    max_budget_allocation: float = Field(
        default=1000000.0, ge=0.0, description="Maximum budget allocation"
    )

    # ==================== DATA PROCESSING ====================
    # Data validation
    missing_data_threshold: float = Field(
        default=0.1, ge=0.0, le=0.5, description="Maximum allowed missing data ratio"
    )
    outlier_detection_method: str = Field(
        default="iqr",
        pattern="^(iqr|zscore|isolation_forest)$",
        description="Outlier detection method",
    )
    outlier_threshold: float = Field(
        default=3.0, ge=1.0, le=5.0, description="Outlier detection threshold"
    )

    # Feature scaling
    scaling_method: str = Field(
        default="standard",
        pattern="^(standard|minmax|robust)$",
        description="Feature scaling method",
    )

    # ==================== LOGGING AND MONITORING ====================
    log_level: str = Field(
        default="INFO",
        pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
        description="Logging level",
    )
    enable_mlflow: bool = Field(default=False, description="Enable MLflow tracking")
    mlflow_tracking_uri: Optional[str] = Field(
        default=None, description="MLflow tracking URI"
    )

    # ==================== API AND EXTERNAL SERVICES ====================
    api_key: Optional[str] = Field(
        default=None, description="API key for external services"
    )
    database_url: Optional[str] = Field(
        default=None, description="Database connection URL"
    )
    cache_ttl: int = Field(
        default=3600, ge=60, le=86400, description="Cache time-to-live in seconds"
    )

    # ==================== PERFORMANCE ====================
    n_jobs: int = Field(
        default=-1, ge=-1, description="Number of parallel jobs (-1 for all cores)"
    )
    random_state: int = Field(
        default=42, ge=0, description="Random state for reproducibility"
    )

    @validator("data_path", "model_path", "output_path", "log_path", pre=True)
    def validate_paths(cls, v):
        """Ensure paths are Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v

    @validator("seasonality_periods")
    def validate_seasonality_periods(cls, v):
        """Ensure seasonality periods are positive integers."""
        if not all(isinstance(p, int) and p > 0 for p in v):
            raise ValueError("All seasonality periods must be positive integers")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
