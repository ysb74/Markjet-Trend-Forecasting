# Market Trend Forecasting - Exploratory Data Analysis
# This file contains all the code cells for comprehensive EDA
# Convert to Jupyter notebook by adding # %% cell markers

# %% [markdown]
# # Market Trend Forecasting - Exploratory Data Analysis
# 
# This notebook provides a comprehensive exploratory data analysis (EDA) of the real estate market data used in our forecasting models.
# 
# ## Table of Contents
# 1. [Data Loading and Overview](#data-loading)
# 2. [Data Quality Assessment](#data-quality)
# 3. [Univariate Analysis](#univariate)
# 4. [Bivariate Analysis](#bivariate)
# 5. [Time Series Analysis](#time-series)
# 6. [Feature Engineering Insights](#feature-engineering)
# 7. [Advanced Visualizations](#advanced-viz)
# 8. [Statistical Analysis](#statistical)
# 9. [Market Insights](#market-insights)
# 10. [Conclusions and Recommendations](#conclusions)

# %% [code]
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, timedelta
from scipy import stats
warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Import custom modules
import sys
sys.path.append('../')

from src.data.ingestion import load_crm_data_live
from src.data.validation import DataQualityReport, TENANT_SCHEMA
from src.utils.logger import get_logger

logger = get_logger(__name__)

print("📊 Libraries imported successfully!")

# %% [markdown]
# ## 1. Data Loading and Overview {#data-loading}

# %% [code]
# Load the data
df = load_crm_data_live()

# Basic information about the dataset
print("📋 Dataset Overview:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\n📊 First few rows:")
print(df.head())

# %% [code]
# Data types and basic statistics
print("🔍 Data Types:")
print(df.dtypes)
print("\n📈 Basic Statistics:")
print(df.describe())

# %% [markdown]
# ## 2. Data Quality Assessment {#data-quality}

# %% [code]
# Generate comprehensive data quality report
quality_report = DataQualityReport(df)
quality_report.generate_report()
quality_report.print_summary()

# %% [code]
# Visualize missing data
missing_data = df.isnull().sum()
missing_data = missing_data[missing_data > 0]

if len(missing_data) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    missing_data.plot(kind='bar', ax=ax)
    plt.title('Missing Data by Column')
    plt.ylabel('Number of Missing Values')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("✅ No missing data found!")

# %% [markdown]
# ## 3. Univariate Analysis {#univariate}

# %% [code]
# Analyze numerical variables
numerical_cols = df.select_dtypes(include=[np.number]).columns

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for i, col in enumerate(numerical_cols[:4]):
    if i < len(axes):
        df[col].hist(bins=30, ax=axes[i], alpha=0.7)
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# %% [code]
# Box plots for outlier detection
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for i, col in enumerate(numerical_cols[:4]):
    if i < len(axes):
        df.boxplot(column=col, ax=axes[i])
        axes[i].set_title(f'Box Plot of {col}')

plt.tight_layout()
plt.show()

# %% [code]
# Analyze categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    if df[col].nunique() < 20:  # Only show if not too many categories
        plt.figure(figsize=(10, 6))
        value_counts = df[col].value_counts()
        plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
        plt.title(f'Distribution of {col}')
        plt.show()

# %% [markdown]
# ## 4. Bivariate Analysis {#bivariate}

# %% [code]
# Correlation analysis
correlation_matrix = df.select_dtypes(include=[np.number]).corr()

plt.figure(figsize=(12, 8))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": .5})
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.show()

# %% [code]
# Scatter plots for key relationships
if 'monthly_rent' in df.columns and 'payment_delays_last_6_months' in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Rent vs Payment Delays
    axes[0].scatter(df['monthly_rent'], df['payment_delays_last_6_months'], alpha=0.6)
    axes[0].set_xlabel('Monthly Rent')
    axes[0].set_ylabel('Payment Delays (Last 6 Months)')
    axes[0].set_title('Rent vs Payment Delays')
    
    # Rent vs Maintenance Requests
    if 'maintenance_requests_last_year' in df.columns:
        axes[1].scatter(df['monthly_rent'], df['maintenance_requests_last_year'], alpha=0.6)
        axes[1].set_xlabel('Monthly Rent')
        axes[1].set_ylabel('Maintenance Requests (Last Year)')
        axes[1].set_title('Rent vs Maintenance Requests')
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 5. Time Series Analysis {#time-series}

# %% [code]
# Convert date columns to datetime if they exist
date_columns = ['lease_start_date', 'lease_end_date']
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Time series analysis of lease starts
if 'lease_start_date' in df.columns:
    # Group by month
    monthly_leases = df.groupby(df['lease_start_date'].dt.to_period('M')).size()
    
    plt.figure(figsize=(12, 6))
    monthly_leases.plot(kind='line', marker='o')
    plt.title('Number of Lease Starts by Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Leases')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Average rent over time
    if 'monthly_rent' in df.columns:
        monthly_rent = df.groupby(df['lease_start_date'].dt.to_period('M'))['monthly_rent'].mean()
        
        plt.figure(figsize=(12, 6))
        monthly_rent.plot(kind='line', marker='o', color='green')
        plt.title('Average Monthly Rent Over Time')
        plt.xlabel('Month')
        plt.ylabel('Average Rent ($)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ## 6. Feature Engineering Insights {#feature-engineering}

# %% [code]
# Create derived features for analysis
if 'lease_start_date' in df.columns and 'lease_end_date' in df.columns:
    df['lease_duration_days'] = (df['lease_end_date'] - df['lease_start_date']).dt.days
    
    plt.figure(figsize=(10, 6))
    df['lease_duration_days'].hist(bins=30, alpha=0.7)
    plt.title('Distribution of Lease Duration (Days)')
    plt.xlabel('Lease Duration (Days)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.show()

# Risk scoring example
if all(col in df.columns for col in ['payment_delays_last_6_months', 'maintenance_requests_last_year']):
    # Create a simple risk score
    df['risk_score'] = (df['payment_delays_last_6_months'] * 0.6 + 
                       df['maintenance_requests_last_year'] * 0.4)
    
    plt.figure(figsize=(10, 6))
    df['risk_score'].hist(bins=30, alpha=0.7, color='red')
    plt.title('Distribution of Risk Score')
    plt.xlabel('Risk Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Risk score vs renewal
    if 'lease_renewal' in df.columns:
        plt.figure(figsize=(10, 6))
        df.boxplot(column='risk_score', by='lease_renewal')
        plt.title('Risk Score by Lease Renewal Status')
        plt.suptitle('')  # Remove automatic title
        plt.xlabel('Lease Renewal (0=Churned, 1=Renewed)')
        plt.ylabel('Risk Score')
        plt.show()

# %% [markdown]
# ## 7. Advanced Visualizations with Plotly {#advanced-viz}

# %% [code]
# Interactive scatter plot
if all(col in df.columns for col in ['monthly_rent', 'payment_delays_last_6_months', 'lease_renewal']):
    fig = px.scatter(df, x='monthly_rent', y='payment_delays_last_6_months', 
                     color='lease_renewal', 
                     title='Rent vs Payment Delays (Colored by Renewal Status)',
                     labels={'lease_renewal': 'Lease Renewal'})
    fig.show()

# 3D scatter plot
if all(col in df.columns for col in ['monthly_rent', 'payment_delays_last_6_months', 'maintenance_requests_last_year']):
    fig = px.scatter_3d(df, x='monthly_rent', y='payment_delays_last_6_months', 
                        z='maintenance_requests_last_year',
                        color='lease_renewal' if 'lease_renewal' in df.columns else None,
                        title='3D Scatter Plot of Key Features')
    fig.show()

# %% [markdown]
# ## 8. Statistical Analysis {#statistical}

# %% [code]
# Statistical summary by groups
if 'lease_renewal' in df.columns:
    print("📊 Statistical Summary by Lease Renewal Status:")
    print("=" * 50)
    
    for col in numerical_cols:
        if col != 'lease_renewal':
            print(f"\n{col}:")
            print(df.groupby('lease_renewal')[col].describe())
            
            # T-test for significant differences
            churned = df[df['lease_renewal'] == 0][col].dropna()
            renewed = df[df['lease_renewal'] == 1][col].dropna()
            
            if len(churned) > 0 and len(renewed) > 0:
                t_stat, p_value = stats.ttest_ind(churned, renewed)
                print(f"T-test p-value: {p_value:.4f}")
                if p_value < 0.05:
                    print("✅ Significant difference between groups")
                else:
                    print("❌ No significant difference between groups")

# %% [markdown]
# ## 9. Market Insights Analysis {#market-insights}

# %% [code]
# Market insights and KPIs
print("🏘️ Market Insights and KPIs")
print("=" * 40)

# Overall statistics
if 'monthly_rent' in df.columns:
    print(f"\n💰 Rent Statistics:")
    print(f"Average Rent: ${df['monthly_rent'].mean():,.2f}")
    print(f"Median Rent: ${df['monthly_rent'].median():,.2f}")
    print(f"Rent Range: ${df['monthly_rent'].min():,.2f} - ${df['monthly_rent'].max():,.2f}")
    print(f"Rent Standard Deviation: ${df['monthly_rent'].std():,.2f}")

if 'lease_renewal' in df.columns:
    renewal_rate = (df['lease_renewal'].sum() / len(df)) * 100
    print(f"\n📈 Renewal Rate: {renewal_rate:.1f}%")
    print(f"Churn Rate: {100 - renewal_rate:.1f}%")

if 'payment_delays_last_6_months' in df.columns:
    avg_delays = df['payment_delays_last_6_months'].mean()
    print(f"\n⏰ Average Payment Delays (6 months): {avg_delays:.2f}")

if 'maintenance_requests_last_year' in df.columns:
    avg_maintenance = df['maintenance_requests_last_year'].mean()
    print(f"🔧 Average Maintenance Requests (year): {avg_maintenance:.2f}")

# Property insights
if 'property_id' in df.columns:
    unique_properties = df['property_id'].nunique()
    total_tenants = len(df)
    avg_tenants_per_property = total_tenants / unique_properties
    
    print(f"\n🏠 Property Insights:")
    print(f"Total Properties: {unique_properties}")
    print(f"Total Tenants: {total_tenants}")
    print(f"Average Tenants per Property: {avg_tenants_per_property:.2f}")

# %% [markdown]
# ## 10. Conclusions and Recommendations {#conclusions}
# 
# Based on the exploratory data analysis, here are the key findings and recommendations:
# 
# ### Key Findings:
# 1. **Data Quality**: [Add observations about data quality]
# 2. **Distribution Patterns**: [Add observations about feature distributions]
# 3. **Correlations**: [Add observations about feature relationships]
# 4. **Time Trends**: [Add observations about temporal patterns]
# 
# ### Recommendations for Modeling:
# 1. **Feature Engineering**: [Suggest new features to create]
# 2. **Data Preprocessing**: [Suggest preprocessing steps]
# 3. **Model Selection**: [Suggest appropriate models based on data characteristics]
# 4. **Validation Strategy**: [Suggest cross-validation approach]
# 
# ### Next Steps:
# 1. Implement the identified preprocessing steps
# 2. Create the suggested engineered features
# 3. Proceed with model training and evaluation
# 4. Set up monitoring for the identified data quality issues

# %% [code]
# Save processed data for modeling
output_path = '../data/processed/eda_insights.csv'
df.to_csv(output_path, index=False)
print(f"💾 Data with EDA insights saved to {output_path}")

# Save summary statistics
summary_stats = df.describe()
summary_stats.to_csv('../data/processed/summary_statistics.csv')
print("📊 Summary statistics saved")

# Save correlation matrix
correlation_matrix.to_csv('../data/processed/correlation_matrix.csv')
print("🔗 Correlation matrix saved")

print("\n✅ Exploratory Data Analysis completed!")
print("\n📋 Next Steps:")
print("1. Review the findings and insights above")
print("2. Proceed to feature engineering notebook")
print("3. Run model training pipeline")
print("4. Evaluate model performance") 