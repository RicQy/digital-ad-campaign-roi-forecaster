# Digital Ad Campaign ROI Forecaster

## Introduction
The Digital Ad Campaign ROI Forecaster is a comprehensive tool designed to forecast and optimize Return on Investment (ROI) for digital advertising campaigns. It supports functionalities like forecasting outcomes based on historical data, optimizing budget allocation, and generating detailed reports.

## Installation

To install the package, ensure you have Python 3.8 or higher. Use pip for the following installation:

```bash
pip install .
```

Alternatively, if you prefer uv:

```bash
uv install
```

## Quick Start

Here's how to get started quickly:

1. **Forecast ROI:**

   ```bash
   python -m ad_roi_forecaster.cli.cli forecast --input data.csv --output forecasts.csv
   ```

2. **Optimize Budget Allocation:**

   ```bash
   python -m ad_roi_forecaster.cli.cli optimize --budget 10000 --start-date 2024-01-01 --end-date 2024-01-31
   ```

3. **Generate Reports:**

   ```bash
   python -m ad_roi_forecaster.cli.cli report --input data.csv --output-dir reports/
   ```

## Example CLI Commands
- **Forecast:** Predict future campaign performance.
  ```
  ad-roi-forecaster forecast -i sample_data/campaign_data.csv -o output/future_forecast.csv
  ```

- **Optimization:** Optimize budget using historical spending data.
  ```
  ad-roi-forecaster optimize -b 5000 -s 2024-01-01 -e 2024-01-31
  ```

- **Reporting:** Generate visual reports with analyses.
  ```
  ad-roi-forecaster report -i sample_data/campaign_data.csv -o output/reports/ -f html
  ```

## Architecture

The application follows a modular architecture with clear separation of concerns. For a detailed view of the system architecture, see:

- **Text-based diagram**: [docs/architecture.md](docs/architecture.md)
- **Visual diagram**: Run `python docs/architecture_diagram.py` to generate a visual representation

## Documentation

For more comprehensive usage details, see [docs/USAGE.md](docs/USAGE.md).

