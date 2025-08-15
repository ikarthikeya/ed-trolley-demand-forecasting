# Emergency Department Trolley Demand Forecasting

## Project Overview

This project implements a comprehensive forecasting system for Emergency Department trolley demand across Irish HSE (Health Service Executive) regions. The system uses machine learning and time series analysis to predict trolley demand up to 14 days in advance, helping healthcare administrators with capacity planning and resource allocation.

## Project Structure

```
dissertaion-backup/
├── src/                    # Main pipeline components
│   ├── pipeline.py         # Main orchestration script
│   ├── data_processor.py   # Data loading and feature engineering
│   ├── models.py          # Model implementations and training
│   ├── evaluator.py       # Model evaluation and explainability
│   ├── visualizer.py      # Plotting and visualization
│   ├── requirements.txt   # Python dependencies
│   └── results/           # Generated outputs
├── data/                   # Input datasets
│   ├── master_trolley_data.csv
│   └── public_holidays_ie.csv
├── comprehensive_eda_generator.py    # Exploratory data analysis
├── regional_champion_analysis.py    # Regional model analysis
└── README.md              # This file
```

## Models Implemented

The system trains and evaluates 7 different forecasting models for each region:

1. **Simple XGBoost** - Gradient boosting with default parameters
2. **Optimized XGBoost** - Enhanced gradient boosting with tuned hyperparameters
3. **LightGBM** - Microsoft's fast gradient boosting implementation
4. **SARIMA** - Seasonal AutoRegressive Integrated Moving Average for time series
5. **Prophet** - Facebook's time series forecasting model
6. **Ensemble** - Weighted combination of the best performing models
7. **Baseline Moving Average** - 14-day moving average for comparison

## Regional Coverage

The system models 6 HSE regions independently:

- HSE Dublin and Midlands
- HSE Dublin and North East  
- HSE Dublin and South East
- HSE Mid West
- HSE South West
- HSE West and North West

Each region has its own champion model selected based on the lowest Mean Absolute Error (MAE) on test data.

## Getting Started

### Prerequisites

Python 3.8 or higher is required. Install the necessary dependencies:

```bash
cd src/
pip install -r requirements.txt
```

### Running the Complete Pipeline

To execute the full forecasting pipeline:

```bash
python src/pipeline.py
```

This will:
- Process the raw data and engineer features
- Train all 7 models for each of the 6 regions (42 models total)
- Evaluate model performance and select regional champions
- Generate SHAP explainability analysis
- Create comprehensive visualizations
- Output 14-day forecasts for each region

### Running Individual Analyses

For exploratory data analysis:
```bash
python comprehensive_eda_generator.py
```

For detailed regional champion analysis:
```bash
python regional_champion_analysis.py
```

## Pipeline Components

### 1. Data Processing
The data processor handles:
- Loading trolley demand data and Irish public holiday information
- Feature engineering including temporal, weather, and holiday features
- Data validation and cleaning
- Temporal train/test splitting to prevent data leakage

### 2. Model Training
The training component:
- Trains 7 different models for each of the 6 HSE regions
- Performs hyperparameter optimization where applicable
- Selects champion models based on Mean Absolute Error performance
- Saves trained models for future use

### 3. Model Evaluation
The evaluation system provides:
- Performance metrics including MAE, RMSE, R-squared, and MAPE
- SHAP (SHapley Additive exPlanations) analysis for model interpretability
- Statistical significance testing between models
- Multi-horizon forecasting evaluation

### 4. Visualization and Reporting
The visualization component generates:
- Time series plots showing historical data and forecasts
- Model performance comparison charts
- Regional analysis dashboards
- Feature importance plots
- Comprehensive summary reports

## Key Features

- **Comprehensive Modeling**: 7 different algorithms trained per region
- **Regional Specialization**: Independent models for each HSE region
- **No Data Leakage**: Proper temporal splitting ensures realistic evaluation
- **Explainable AI**: SHAP analysis provides model interpretability
- **Automated Pipeline**: Single command execution for complete analysis
- **Extensive Evaluation**: Multiple metrics and statistical testing
- **Professional Visualizations**: Publication-ready charts and dashboards

## Output Results

The pipeline generates several categories of outputs:

### Data Outputs
- Processed and cleaned datasets
- Feature engineering summaries
- Data quality reports

### Model Outputs
- Trained models for all regions and algorithms
- Champion model selections
- Model comparison statistics
- 14-day regional forecasts

### Analysis Outputs
- Exploratory data analysis visualizations
- Feature importance rankings
- SHAP explainability plots
- Performance evaluation metrics

### Reports
- Comprehensive analysis dashboards
- Regional forecast summaries
- Model performance comparisons

## Usage Examples

### Running Regional Analysis
```python
from src.models import RegionalModelTrainer

trainer = RegionalModelTrainer("results/models")
regional_data = trainer.prepare_regional_data(data)
results = trainer.train_regional_models(regional_data)
forecasts = trainer.generate_regional_forecasts(regional_data)
```

### Training Individual Models
```python
from src.models import ModelTrainer

trainer = ModelTrainer("results/models")
results = trainer.train_models(X_train, y_train, X_test, y_test)
champion = trainer.champion_model
```

### Data Processing
```python
from src.data_processor import DataProcessor

processor = DataProcessor()
processed_data = processor.process_data("data/master_trolley_data.csv")
X_train, X_test, y_train, y_test = processor.prepare_train_test_split(processed_data)
```

## Model Performance

Current champion models by region:
- **HSE Dublin & Midlands**: Ensemble (MAE: 4.447)
- **HSE Dublin & North East**: SARIMA (MAE: 6.653)  
- **HSE Dublin & South East**: Optimized XGBoost (MAE: 4.897)
- **HSE Mid West**: Ensemble (MAE: 7.992)
- **HSE South West**: Ensemble (MAE: 7.364)
- **HSE West & North West**: Ensemble (MAE: 6.353)

Overall champion across all data: Simple XGBoost (MAE: 0.081, R²: 1.000)

## Dependencies

The project requires the following Python packages:

**Core Libraries:**
- pandas - Data manipulation and analysis
- numpy - Numerical computing
- scikit-learn - Machine learning algorithms

**Machine Learning Models:**
- xgboost - Gradient boosting framework
- lightgbm - Microsoft's gradient boosting
- statsmodels - Statistical modeling (for SARIMA)
- prophet - Facebook's time series forecasting

**Analysis and Visualization:**
- shap - Model explainability
- matplotlib - Basic plotting
- seaborn - Statistical visualization

**Data Processing:**
- datetime - Date and time handling
- pathlib - File system paths

All dependencies are listed in `src/requirements.txt` and can be installed using pip.

## Data Sources

The system uses two main datasets:

1. **master_trolley_data.csv** - Historical trolley demand data across Irish hospitals
2. **public_holidays_ie.csv** - Irish public holiday calendar for feature engineering

## Project Status

This project is complete and operational. The pipeline has been tested and validated across all HSE regions with comprehensive evaluation metrics. All components are working as intended and ready for operational use in healthcare capacity planning.
# ed-trolley-demand-forecasting
