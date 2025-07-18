# Digital Ad Campaign ROI Forecaster Documentation

This directory contains comprehensive documentation for the Digital Ad Campaign ROI Forecaster application.

## Documentation Files

### 📖 [USAGE.md](USAGE.md)
**Complete usage guide covering:**
- Advanced configuration options
- Model assumptions and selection
- Interpretation guidelines for forecasts and optimizations
- Data format requirements
- Troubleshooting guide

### 🏗️ [architecture.md](architecture.md)
**System architecture documentation:**
- Component overview with ASCII diagrams
- Data flow descriptions
- Module responsibilities
- Design principles

### 📊 [architecture_diagram.py](architecture_diagram.py)
**Visual architecture diagram generator:**
- Python script to generate visual system architecture
- Uses matplotlib for diagram creation
- Run with: `python docs/architecture_diagram.py`

## Quick Reference

### Key CLI Commands

#### Forecasting
```bash
# Basic forecast
ad-roi-forecaster forecast -i data.csv -o forecast.csv

# Advanced forecast with specific model
ad-roi-forecaster forecast -i data.csv -o forecast.csv --model arima --periods 60 --confidence 0.99
```

#### Optimization
```bash
# Budget optimization
ad-roi-forecaster optimize -b 10000 -s 2024-01-01 -e 2024-01-31

# Optimization with data input
ad-roi-forecaster optimize -i data.csv -b 5000 -o allocation.csv
```

#### Reporting
```bash
# Generate HTML report
ad-roi-forecaster report -i data.csv -o reports/ -f html

# Generate comprehensive report with forecasts
ad-roi-forecaster report -i data.csv -o reports/ -f html -f pdf --include-forecasts --include-optimization
```

### Data Format

Your CSV files should contain these columns:
- `campaign_id` (required)
- `date` (required, YYYY-MM-DD format)
- `spend` (required)
- `impressions` (required)
- `clicks` (required)
- `conversions` (required)
- `revenue` (required)
- Additional optional columns for enhanced analysis

### Model Types

1. **Auto**: Automatically selects best model (default)
2. **Baseline**: Linear regression model
3. **ARIMA**: Time series forecasting
4. **Seasonal Naive**: Seasonal pattern-based forecasting

### Common Use Cases

#### 1. Performance Forecasting
- Predict future campaign performance
- Estimate ROI for different spend scenarios
- Plan budget allocation for upcoming periods

#### 2. Budget Optimization
- Maximize ROI across multiple campaigns
- Optimize allocation within budget constraints
- Identify underperforming campaigns

#### 3. Performance Analysis
- Generate comprehensive performance reports
- Compare campaigns across platforms
- Identify trends and patterns

## Configuration

### Environment Variables
Create a `.env` file in your project root:
```env
LOG_LEVEL=INFO
FORECAST_HORIZON=30
CONFIDENCE_INTERVAL=0.95
MAX_BUDGET_ALLOCATION=1000000.0
```

### Settings File
The application uses `config/settings.py` for configuration management with Pydantic validation.

## Getting Help

1. **CLI Help**: Use `ad-roi-forecaster --help` or `ad-roi-forecaster [command] --help`
2. **Verbose Logging**: Add `--verbose` to any command for detailed output
3. **Documentation**: Review the files in this directory
4. **Configuration**: Check your `.env` file and settings
5. **Data Issues**: Ensure your CSV files match the required format

## Development

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ad_roi_forecaster
```

### Development Setup
```bash
# Install in development mode
pip install -e .

# Install development dependencies
pip install -e .[dev]
```

## Examples

The `sample_data/` directory contains example CSV files that demonstrate the expected data format and can be used for testing the application.

## Support

For technical issues:
1. Check the troubleshooting section in [USAGE.md](USAGE.md)
2. Review the logs in your specified log directory
3. Ensure all dependencies are properly installed
4. Verify your data format matches the requirements
