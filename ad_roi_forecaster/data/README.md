# Data Input & Validation Module

This module provides comprehensive functionality for loading, validating, and processing digital advertising campaign data from CSV files.

## Features

- **CSV Loading**: Flexible CSV reading with automatic column detection and data type coercion
- **Data Validation**: Comprehensive validation including missing columns, date ranges, numeric bounds, and business logic consistency
- **Pydantic Schemas**: Structured data models for campaign records with automatic validation
- **Statistical Analysis**: Outlier detection and data quality assessment

## Main Components

### 1. Data Loading (`loader.py`)

The `CampaignDataLoader` class provides robust CSV loading capabilities:

```python
from ad_roi_forecaster.data import load_campaign_data

# Load single CSV file
df = load_campaign_data("path/to/campaign_data.csv")

# Load multiple CSV files from directory
df = load_campaign_data("path/to/csv_directory/")
```

**Features:**
- Automatic column detection and mapping
- Data type coercion (dates, numbers, strings)
- Flexible column naming support
- Missing value handling
- Data cleaning and preprocessing

### 2. Data Validation (`validator.py`)

The `CampaignDataValidator` provides comprehensive data quality checks:

```python
from ad_roi_forecaster.data import validate_campaign_data

# Validate loaded data
result = validate_campaign_data(df)

print(f"Valid: {result.is_valid}")
print(f"Errors: {result.errors}")
print(f"Warnings: {result.warnings}")
```

**Validation Checks:**
- Required columns presence
- Date range validation
- Numeric bounds checking
- Business logic consistency (clicks ≤ impressions, conversions ≤ clicks)
- Statistical outlier detection
- Data completeness assessment

### 3. Pydantic Schemas (`schemas.py`)

Structured data models for type safety and validation:

```python
from ad_roi_forecaster.data import CampaignRecord, CampaignDataset

# Individual campaign record
record = CampaignRecord(
    campaign_id="CAMP001",
    campaign_name="Summer Sale",
    campaign_type="search",
    platform="google",
    date="2024-01-01",
    spend=150.50,
    impressions=10000,
    clicks=250,
    conversions=15,
    revenue=450.00
)

# Access calculated metrics
print(f"ROI: {record.roi:.2%}")
print(f"ROAS: {record.roas:.2f}")
print(f"CTR: {record.ctr:.2%}")
```

## Data Schema

### Required Columns

The following columns are required in your CSV files:

| Column | Type | Description |
|--------|------|-------------|
| `campaign_id` | string | Unique campaign identifier |
| `campaign_name` | string | Human-readable campaign name |
| `campaign_type` | string | Campaign type (search, display, video, social, shopping) |
| `platform` | string | Advertising platform (google, facebook, instagram, etc.) |
| `date` | datetime | Campaign date |
| `spend` | float | Campaign spend amount |
| `impressions` | int | Number of impressions |
| `clicks` | int | Number of clicks |
| `conversions` | int | Number of conversions |
| `revenue` | float | Revenue generated |

### Optional Columns

| Column | Type | Description |
|--------|------|-------------|
| `target_audience` | string | Target audience segment |
| `geographic_region` | string | Geographic targeting region |
| `budget_daily` | float | Daily budget limit |
| `budget_total` | float | Total campaign budget |

## Example Usage

### Basic Usage

```python
from ad_roi_forecaster.data import load_campaign_data, validate_campaign_data

# Load data
df = load_campaign_data("campaign_data.csv")

# Validate data
validation_result = validate_campaign_data(df)

if validation_result.is_valid:
    print("Data is valid and ready for analysis!")
else:
    print("Data validation failed:")
    for error in validation_result.errors:
        print(f"  - {error}")
```

### Advanced Usage

```python
from ad_roi_forecaster.data import CampaignDataLoader, CampaignDataValidator
from datetime import datetime

# Custom loader with specific settings
loader = CampaignDataLoader()
df = loader.load_csv("data.csv", encoding="utf-8", delimiter=",")

# Custom validation with date range
validator = CampaignDataValidator(df)
result = validator.validate(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    check_outliers=True
)
```

## Column Mapping

The loader automatically detects common column name variations:

- `campaign_id`: campaign_id, Campaign_ID, campaignId, id
- `spend`: spend, Spend, cost, Cost, investment
- `impressions`: impressions, Impressions, impr, Impr
- `clicks`: clicks, Clicks, click, Click
- `revenue`: revenue, Revenue, sales, Sales, value

## Error Handling

The module provides comprehensive error handling:

- **File not found**: Clear error message with file path
- **Invalid CSV format**: Detailed parsing error information
- **Missing columns**: List of missing required columns
- **Data type errors**: Specific column and conversion errors
- **Validation failures**: Detailed validation error messages

## Performance Considerations

- Uses pandas for efficient data processing
- Automatic data type optimization
- Memory-efficient processing for large datasets
- Batch processing support for multiple files

## Testing

Run the test script to verify functionality:

```bash
python test_data_module.py
```

This will:
1. Load sample campaign data
2. Validate the data quality
3. Calculate performance metrics
4. Display results

## Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical operations
- pydantic: Data validation and schemas
- typing: Type hints support

## Future Enhancements

Planned improvements:
- Support for additional file formats (Excel, JSON)
- Advanced anomaly detection algorithms
- Data quality scoring
- Integration with external data sources
- Real-time data validation APIs
