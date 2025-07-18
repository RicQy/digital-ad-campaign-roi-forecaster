# Architecture Diagram

## System Overview

The Digital Ad Campaign ROI Forecaster follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          CLI Interface (ad_roi_forecaster.cli)                  │
│                                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   forecast  │    │   optimize  │    │   report    │    │   --help    │     │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘     │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Data Processing Layer                                    │
│                      (ad_roi_forecaster.data)                                  │
│                                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                       │
│  │   loader.py │    │ validator.py│    │ schemas.py  │                       │
│  │             │    │             │    │             │                       │
│  │ • CSV Load  │    │ • Quality   │    │ • Pydantic  │                       │
│  │ • Parsing   │    │   Checks    │    │   Models    │                       │
│  │ • Format    │    │ • Outliers  │    │ • Validation│                       │
│  └─────────────┘    └─────────────┘    └─────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                      ┌─────────────┼─────────────┐
                      │             │             │
                      ▼             ▼             ▼
┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐
│   Forecasting Engine    │ │  Optimization Engine    │ │  Visualization Engine   │
│  (forecasting module)   │ │  (optimization module)  │ │  (visualization module) │
│                         │ │                         │ │                         │
│ ┌─────────────────────┐ │ │ ┌─────────────────────┐ │ │ ┌─────────────────────┐ │
│ │   Model Selection   │ │ │ │   ROI Optimizer     │ │ │ │     Plots           │ │
│ │                     │ │ │ │                     │ │ │ │                     │ │
│ │ • Auto Selection    │ │ │ │ • Budget Allocation │ │ │ │ • Performance       │ │
│ │ • Cross Validation  │ │ │ │ • Constraint Solver │ │ │ │ • Trends            │ │
│ │ • Model Comparison  │ │ │ │ • ROI Maximization  │ │ │ │ • Comparisons       │ │
│ └─────────────────────┘ │ │ └─────────────────────┘ │ │ └─────────────────────┘ │
│                         │ │                         │ │                         │
│ ┌─────────────────────┐ │ │ ┌─────────────────────┐ │ │ ┌─────────────────────┐ │
│ │   Model Types       │ │ │ │   Constraints       │ │ │ │     Reports         │ │
│ │                     │ │ │ │                     │ │ │ │                     │ │
│ │ • Baseline (LR)     │ │ │ │ • Budget Limits     │ │ │ │ • HTML Reports      │ │
│ │ • ARIMA             │ │ │ │ • Min ROI Threshold │ │ │ │ • PDF Reports       │ │
│ │ • Seasonal Naive    │ │ │ │ • Campaign Filters  │ │ │ │ • Interactive       │ │
│ └─────────────────────┘ │ │ └─────────────────────┘ │ │ └─────────────────────┘ │
└─────────────────────────┘ └─────────────────────────┘ └─────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Configuration Layer                                │
│                               (config module)                                  │
│                                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                       │
│  │ settings.py │    │   .env      │    │ pyproject.  │                       │
│  │             │    │             │    │    toml     │                       │
│  │ • Pydantic  │    │ • Env Vars  │    │             │                       │
│  │   Settings  │    │ • Secrets   │    │ • Metadata  │                       │
│  │ • Validation│    │ • Overrides │    │ • Dependencies │                    │
│  └─────────────┘    └─────────────┘    └─────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Input Processing
```
CSV Data → Data Loader → Validator → Pydantic Schemas → Clean Dataset
```

### 2. Forecasting Workflow
```
Clean Dataset → Feature Engineering → Model Selection → Training → Predictions → Confidence Intervals
```

### 3. Optimization Workflow
```
Historical Data → Performance Modeling → Constraint Setup → Optimization Solver → Allocation Results
```

### 4. Reporting Workflow
```
Results Data → Chart Generation → Report Templates → HTML/PDF Output → File System
```

## Component Responsibilities

### CLI Layer
- **Purpose**: User interface and command routing
- **Components**: `cli.py`, `__main__.py`
- **Responsibilities**:
  - Parse command-line arguments
  - Validate input parameters
  - Route to appropriate modules
  - Display results and progress

### Data Processing Layer
- **Purpose**: Data ingestion and validation
- **Components**: `loader.py`, `validator.py`, `schemas.py`
- **Responsibilities**:
  - Load campaign data from CSV files
  - Validate data quality and format
  - Apply business rules and constraints
  - Convert to internal data structures

### Forecasting Engine
- **Purpose**: Predictive modeling and forecasting
- **Components**: `model_selector.py`, `baseline.py`, `time_series.py`
- **Responsibilities**:
  - Select optimal forecasting model
  - Train models on historical data
  - Generate predictions with confidence intervals
  - Handle different forecasting scenarios

### Optimization Engine
- **Purpose**: Budget allocation optimization
- **Components**: `roi_optimizer.py`
- **Responsibilities**:
  - Model campaign performance curves
  - Solve constrained optimization problems
  - Maximize ROI given budget constraints
  - Handle multiple campaigns and platforms

### Visualization Engine
- **Purpose**: Report generation and visualization
- **Components**: `plots.py`, `report.py`
- **Responsibilities**:
  - Generate performance visualizations
  - Create comprehensive reports
  - Support multiple output formats
  - Provide interactive analysis tools

### Configuration Layer
- **Purpose**: Application configuration management
- **Components**: `settings.py`, `.env`, `pyproject.toml`
- **Responsibilities**:
  - Manage application settings
  - Handle environment variables
  - Validate configuration parameters
  - Provide default values

## Key Design Principles

1. **Modularity**: Each component has a single responsibility
2. **Testability**: All components can be tested independently
3. **Configurability**: Behavior can be customized via configuration
4. **Extensibility**: New models and features can be added easily
5. **Robustness**: Comprehensive error handling and validation
6. **Performance**: Efficient algorithms and parallel processing support
