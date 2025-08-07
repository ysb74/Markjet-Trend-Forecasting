# 📓 Jupyter Notebooks for Market Trend Forecasting

This directory contains comprehensive Jupyter notebooks for the complete machine learning workflow in the Market Trend Forecasting project.

## 📋 Notebook Overview

### 1. **01_exploratory_data_analysis.ipynb** 🔍
**Comprehensive Exploratory Data Analysis**

- **Purpose**: Deep dive into the real estate market data
- **Sections**:
  - Data Loading and Overview
  - Data Quality Assessment
  - Univariate Analysis
  - Bivariate Analysis
  - Time Series Analysis
  - Feature Engineering Insights
  - Advanced Visualizations with Plotly
  - Statistical Analysis
  - Market Insights Analysis
  - Conclusions and Recommendations

- **Key Features**:
  - Automated data quality reports
  - Interactive visualizations
  - Statistical significance testing
  - Market KPI calculations
  - Data export for further analysis

### 2. **02_feature_engineering.ipynb** 🔧
**Advanced Feature Engineering and Selection**

- **Purpose**: Create and optimize features for ML models
- **Sections**:
  - Feature Creation (Time-based, Risk, Financial, Interaction)
  - Feature Selection (Statistical tests, Random Forest importance)
  - Feature Scaling (StandardScaler)
  - Feature Importance Analysis
  - Model Preparation

- **Key Features**:
  - Automated feature creation from raw data
  - Statistical feature selection
  - Feature importance ranking
  - Data scaling and encoding
  - Model-ready dataset preparation

### 3. **03_model_evaluation.ipynb** 🤖
**Model Training, Evaluation, and Deployment**

- **Purpose**: Train, compare, and deploy ML models
- **Sections**:
  - Data Preparation
  - Model Training (Multiple algorithms)
  - Model Evaluation (Metrics comparison)
  - Model Comparison (Cross-validation)
  - Hyperparameter Tuning
  - Feature Importance Analysis
  - Model Deployment Preparation

- **Key Features**:
  - Multiple ML algorithms comparison
  - Comprehensive evaluation metrics
  - Cross-validation analysis
  - Hyperparameter optimization
  - Model deployment automation

## 🚀 Quick Start

### Prerequisites
```bash
# Install required packages
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Start Jupyter Lab
jupyter lab
```

### Running the Notebooks

1. **Start with EDA**:
   ```bash
   jupyter lab notebooks/01_exploratory_data_analysis.ipynb
   ```

2. **Continue with Feature Engineering**:
   ```bash
   jupyter lab notebooks/02_feature_engineering.ipynb
   ```

3. **Finish with Model Evaluation**:
   ```bash
   jupyter lab notebooks/03_model_evaluation.ipynb
   ```

## 📊 Expected Outputs

### From EDA Notebook:
- Data quality reports
- Statistical summaries
- Visualization plots
- Market insights
- Processed data files

### From Feature Engineering Notebook:
- Engineered features
- Feature importance rankings
- Scaled datasets
- Model-ready data
- Feature selection results

### From Model Evaluation Notebook:
- Trained models
- Performance comparisons
- Best model selection
- Deployment-ready models
- Model metadata

## 🔧 Customization

### Adding New Features
1. Edit the feature creation section in `02_feature_engineering.ipynb`
2. Add new feature logic
3. Update feature selection criteria
4. Re-run the notebook

### Adding New Models
1. Edit the model definitions in `03_model_evaluation.ipynb`
2. Add new model to the dictionaries
3. Update evaluation metrics if needed
4. Re-run the notebook

### Custom Visualizations
1. Modify plotting code in any notebook
2. Add new visualization functions
3. Update chart styles and layouts
4. Export custom plots

## 📈 Workflow Integration

These notebooks are designed to work seamlessly with the main project:

- **Data Sources**: Uses `src/data/ingestion.py` for data loading
- **Preprocessing**: Leverages `src/data/preprocessing.py` for data cleaning
- **Models**: Integrates with `src/models/ml_models.py` for training
- **Validation**: Uses `src/data/validation.py` for data quality checks
- **Logging**: Implements `src/utils/logger.py` for consistent logging
- **Configuration**: Uses `src/utils/config.py` for settings management

## 🎯 Best Practices

### Notebook Execution
1. **Run cells sequentially** - Don't skip cells
2. **Check outputs** - Verify each step produces expected results
3. **Save frequently** - Save notebooks after major sections
4. **Document findings** - Add notes and observations

### Data Handling
1. **Backup original data** - Never modify raw data
2. **Version control** - Track changes to processed data
3. **Document transformations** - Record all data modifications
4. **Validate results** - Check for data quality issues

### Model Development
1. **Start simple** - Begin with basic models
2. **Iterate gradually** - Add complexity step by step
3. **Cross-validate** - Always use cross-validation
4. **Monitor performance** - Track metrics over time

## 🐛 Troubleshooting

### Common Issues

**Import Errors**:
```python
# Add project root to path
import sys
sys.path.append('../')
```

**Missing Dependencies**:
```bash
pip install -r requirements.txt
```

**Data Loading Issues**:
- Check file paths
- Verify data format
- Ensure required columns exist

**Model Training Errors**:
- Check data quality
- Verify feature scaling
- Ensure target variable exists

### Getting Help

1. **Check logs** - Review error messages carefully
2. **Validate data** - Ensure data is in expected format
3. **Test components** - Run individual functions separately
4. **Review documentation** - Check function docstrings

## 📚 Additional Resources

- **Project Documentation**: See main `README.md`
- **API Documentation**: Check `api/` directory
- **Configuration**: Review `config/config.yaml`
- **Tests**: Run `python -m pytest tests/`

## 🤝 Contributing

When adding new notebooks:

1. **Follow naming convention**: `XX_descriptive_name.ipynb`
2. **Include markdown sections**: Document each step
3. **Add to conversion script**: Update `scripts/convert_to_notebook.py`
4. **Test thoroughly**: Ensure all cells run successfully
5. **Update this README**: Add notebook description

---

**Happy Analyzing! 🎉** 