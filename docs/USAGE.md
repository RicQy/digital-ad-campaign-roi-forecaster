# Digital Ad Campaign ROI Forecaster - Usage Guide

## Table of Contents
1. [Advanced Configuration](#advanced-configuration)
2. [Model Assumptions](#model-assumptions)
3. [Interpretation Guidelines](#interpretation-guidelines)
4. [Data Format Requirements](#data-format-requirements)
5. [Troubleshooting](#troubleshooting)

## Advanced Configuration

### Configuration Files

The application uses several configuration options that can be set via environment variables or configuration files:

#### Environment Variables
Create a `.env` file in your project root with the following options:

```env
# Paths
DATA_PATH=./sample_data
MODEL_PATH=./models
OUTPUT_PATH=./outputs
LOG_PATH=./logs

# Model Hyperparameters
LEARNING_RATE=0.001
BATCH_SIZE=32
NUM_EPOCHS=100
VALIDATION_SPLIT=0.2

# Forecasting Parameters
FORECAST_HORIZON=30
CONFIDENCE_INTERVAL=0.95
SEASONALITY_PERIODS=[7, 30, 365]

# Feature Engineering
LAG_FEATURES=7
ROLLING_WINDOW_SIZE=7

# Optimization Constraints
MAX_ITERATIONS=1000
CONVERGENCE_THRESHOLD=1e-6
MAX_TRAINING_TIME=3600

# Budget and ROI Constraints
MIN_ROI_THRESHOLD=1.0
MAX_BUDGET_ALLOCATION=1000000.0

# Data Processing
MISSING_DATA_THRESHOLD=0.1
OUTLIER_DETECTION_METHOD=iqr
OUTLIER_THRESHOLD=3.0
SCALING_METHOD=standard

# Logging and Monitoring
LOG_LEVEL=INFO
ENABLE_MLFLOW=false

# Performance
N_JOBS=-1
RANDOM_STATE=42
```

### CLI Command Options

#### Forecast Command
```bash
ad-roi-forecaster forecast [OPTIONS]
```

**Options:**
- `--input, -i`: Path to input CSV file containing campaign data (required)
- `--output, -o`: Path to output CSV file for forecasts (required)
- `--periods, -p`: Number of periods to forecast (default: 30)
- `--spend-scenario, -s`: Spend scenario for forecasting (optional)
- `--model, -m`: Model type (auto, baseline, arima, seasonal_naive) (default: auto)
- `--confidence, -c`: Confidence level for predictions (default: 0.95)
- `--cv-folds`: Cross-validation folds for model selection (default: 5)
- `--scoring`: Scoring metric (mae, mse, rmse, mape, r2_score) (default: mae)
- `--verbose, -v`: Enable verbose logging

#### Optimize Command
```bash
ad-roi-forecaster optimize [OPTIONS]
```

**Options:**
- `--input, -i`: Path to input CSV file (optional)
- `--budget, -b`: Total budget for optimization (required)
- `--start-date, -s`: Start date for optimization period (YYYY-MM-DD format)
- `--end-date, -e`: End date for optimization period (YYYY-MM-DD format)
- `--output, -o`: Path to output CSV file for allocation results
- `--campaigns, -c`: Specific campaign IDs to optimize (repeatable)
- `--platforms, -p`: Specific platforms to optimize (repeatable)

#### Report Command
```bash
ad-roi-forecaster report [OPTIONS]
```

**Options:**
- `--input, -i`: Path to input CSV file (required)
- `--output-dir, -o`: Directory to save generated reports (required)
- `--format, -f`: Report format (html, pdf) (default: html, repeatable)
- `--title, -t`: Report title (default: "Digital Ad Campaign ROI Analysis")
- `--company, -c`: Company name for report (default: "Campaign Analytics")
- `--include-forecasts`: Include forecasting analysis in report
- `--forecast-periods`: Number of periods to forecast for report (default: 30)
- `--include-optimization`: Include budget optimization analysis in report
- `--optimization-budget`: Budget for optimization analysis
- `--dpi`: DPI for plot images (default: 300)

## Model Assumptions

### Forecasting Models

The application supports multiple forecasting models with different assumptions:

#### 1. Baseline Linear Regression Model
**Assumptions:**
- Linear relationship between features (spend, impressions, clicks) and conversions
- Features are statistically independent
- Residuals are normally distributed
- No significant seasonality or trends beyond what's captured by features

**Best for:** Short-term forecasts with stable spending patterns

#### 2. ARIMA (AutoRegressive Integrated Moving Average)
**Assumptions:**
- Time series is stationary after differencing
- Seasonal patterns can be captured through seasonal differencing
- Residuals are white noise
- No structural breaks in the data

**Best for:** Time series with clear trends and seasonal patterns

#### 3. Seasonal Naive
**Assumptions:**
- Future values will repeat the pattern from the same season in the previous cycle
- Seasonality is the primary driver of performance
- No underlying trend beyond seasonal patterns

**Best for:** Highly seasonal campaigns with consistent patterns

#### 4. Auto Model Selection
**Process:**
- Evaluates all available models using cross-validation
- Selects the model with the best performance on the specified scoring metric
- Considers both statistical significance and practical relevance

### Budget Optimization Model

**Assumptions:**
- Diminishing returns on ad spend (concave utility function)
- Historical performance is indicative of future performance
- Budget constraints are hard constraints
- ROI can be modeled as a function of spend per campaign
- No interaction effects between campaigns

## Interpretation Guidelines

### Forecasting Results

#### Key Metrics to Examine:
1. **Predicted Conversions**: The forecasted number of conversions
2. **Confidence Intervals**: Range of likely outcomes (default: 95% confidence)
3. **Model Performance**: Cross-validation scores and residual analysis

#### Interpretation Tips:
- **Wider confidence intervals** indicate higher uncertainty
- **Consistent patterns** suggest the model captures underlying trends well
- **Residual analysis** should show no systematic patterns

### Budget Optimization Results

#### Key Outputs:
1. **Optimized Allocation**: Recommended budget distribution across campaigns
2. **Expected ROI**: Predicted ROI from optimized allocation
3. **Allocation Changes**: Difference from current to optimized spending

#### Interpretation Guidelines:
- **Large allocation changes** may indicate significant optimization opportunities
- **Small changes** suggest current allocation is near-optimal
- **ROI improvements** should be evaluated against implementation costs

### Report Analysis

#### Performance Metrics:
- **ROI (Return on Investment)**: (Revenue - Spend) / Spend × 100
- **ROAS (Return on Ad Spend)**: Revenue / Spend
- **CTR (Click-Through Rate)**: Clicks / Impressions × 100
- **Conversion Rate**: Conversions / Clicks × 100
- **CPC (Cost Per Click)**: Spend / Clicks
- **CPA (Cost Per Acquisition)**: Spend / Conversions

#### Benchmarking:
- Compare performance across campaigns, platforms, and time periods
- Identify top-performing segments for scaling
- Identify underperforming segments for optimization or elimination

## Data Format Requirements

### Input CSV Structure
Your input CSV file must contain the following columns:

```csv
campaign_id,campaign_name,campaign_type,platform,date,spend,impressions,clicks,conversions,revenue,target_audience,geographic_region,budget_daily,budget_total
```

#### Required Columns:
- `campaign_id`: Unique identifier for each campaign
- `date`: Date in YYYY-MM-DD format
- `spend`: Amount spent on the campaign
- `impressions`: Number of ad impressions
- `clicks`: Number of clicks received
- `conversions`: Number of conversions/acquisitions
- `revenue`: Revenue generated from the campaign

#### Optional Columns:
- `campaign_name`: Human-readable campaign name
- `campaign_type`: Type of campaign (search, display, video, social, shopping)
- `platform`: Advertising platform (google, facebook, instagram, etc.)
- `target_audience`: Target audience description
- `geographic_region`: Geographic targeting
- `budget_daily`: Daily budget limit
- `budget_total`: Total campaign budget

### Data Quality Requirements
- **No missing values** in required columns
- **Positive values** for spend, impressions, clicks, conversions, revenue
- **Clicks ≤ Impressions** (logical constraint)
- **Conversions ≤ Clicks** (logical constraint)
- **Consistent date format** (YYYY-MM-DD)

## Troubleshooting

### Common Issues and Solutions

#### 1. Data Loading Errors
**Problem:** CSV file cannot be loaded
**Solutions:**
- Check file path and permissions
- Verify CSV format and column names
- Ensure proper encoding (UTF-8)

#### 2. Validation Errors
**Problem:** Data validation fails
**Solutions:**
- Review data quality requirements
- Check for missing values or negative numbers
- Verify logical constraints (clicks ≤ impressions)

#### 3. Model Selection Issues
**Problem:** Auto model selection fails
**Solutions:**
- Ensure sufficient data (minimum 30 data points)
- Check for extreme outliers
- Try specific model selection instead of auto

#### 4. Optimization Problems
**Problem:** Budget optimization doesn't converge
**Solutions:**
- Increase maximum iterations
- Check budget constraints
- Verify data quality and sufficient historical performance

#### 5. Report Generation Failures
**Problem:** Report generation fails
**Solutions:**
- Ensure output directory exists and is writable
- Check for sufficient disk space
- Verify all required dependencies are installed

### Performance Optimization
- **Large datasets**: Use `--n-jobs` parameter for parallel processing
- **Memory issues**: Process data in chunks or reduce batch size
- **Slow optimization**: Reduce convergence threshold or maximum iterations

### Getting Help
For additional support:
1. Check the logs in the specified log directory
2. Enable verbose logging with `--verbose` flag
3. Review the configuration settings in your `.env` file
4. Consult the source code documentation for advanced customization
