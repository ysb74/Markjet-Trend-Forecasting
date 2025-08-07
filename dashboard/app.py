"""Streamlit dashboard for Market Trend Forecasting visualization and interaction."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import joblib
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import config
from src.utils.logger import get_logger
from src.data.validation import DataQualityReport
from src.models.time_series import train_time_series_models

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Market Trend Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .alert-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .alert-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_sample_data():
    """Load sample data for demonstration."""
    np.random.seed(42)
    
    # Generate sample tenant data
    date_range = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
    n_samples = 1000
    
    data = {
        'tenant_id': np.arange(1, n_samples + 1),
        'property_id': [f'P{np.random.randint(1, 100):03d}' for _ in range(n_samples)],
        'lease_start_date': np.random.choice(date_range[:-365], n_samples),
        'monthly_rent': np.random.normal(1800, 400, n_samples).clip(800, 5000),
        'payment_delays_last_6_months': np.random.poisson(0.5, n_samples),
        'maintenance_requests_last_year': np.random.poisson(2, n_samples),
        'feedback_score': np.random.beta(4, 1, n_samples) * 5,
        'status': np.random.choice(['Active', 'Inactive'], n_samples, p=[0.8, 0.2]),
        'lease_renewal': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'beds': np.random.choice([1, 2, 3, 4], n_samples, p=[0.2, 0.4, 0.3, 0.1]),
        'baths': np.random.choice([1, 1.5, 2, 2.5, 3], n_samples, p=[0.2, 0.2, 0.4, 0.15, 0.05]),
        'sqft': np.random.normal(1200, 300, n_samples).clip(500, 3000),
    }
    
    df = pd.DataFrame(data)
    df['lease_end_date'] = df['lease_start_date'] + pd.Timedelta(days=365)
    df['monthly_rent'] = df['monthly_rent'].round(2)
    df['sqft'] = df['sqft'].astype(int)
    
    return df

@st.cache_data
def get_data():
    """Get data with caching."""
    return load_sample_data()

def show_overview_tab():
    """Display overview dashboard."""
    st.header("📊 Market Overview")
    
    df = get_data()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_properties = df['property_id'].nunique()
        st.metric("Total Properties", f"{total_properties:,}")
    
    with col2:
        total_tenants = len(df)
        st.metric("Total Tenants", f"{total_tenants:,}")
    
    with col3:
        avg_rent = df['monthly_rent'].mean()
        st.metric("Average Rent", f"${avg_rent:,.0f}")
    
    with col4:
        renewal_rate = (df['lease_renewal'].sum() / len(df)) * 100
        st.metric("Renewal Rate", f"{renewal_rate:.1f}%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Rent Distribution")
        fig_rent = px.histogram(
            df, x='monthly_rent', nbins=30,
            title="Distribution of Monthly Rent",
            labels={'monthly_rent': 'Monthly Rent ($)', 'count': 'Frequency'}
        )
        fig_rent.update_layout(showlegend=False)
        st.plotly_chart(fig_rent, use_container_width=True)
    
    with col2:
        st.subheader("Property Types")
        property_counts = df['beds'].value_counts().sort_index()
        fig_beds = px.pie(
            values=property_counts.values,
            names=[f"{bed} Bed" for bed in property_counts.index],
            title="Properties by Bedroom Count"
        )
        st.plotly_chart(fig_beds, use_container_width=True)
    
    # Time series analysis
    st.subheader("📈 Rent Trends Over Time")
    
    # Aggregate monthly rent data
    df['lease_start_month'] = df['lease_start_date'].dt.to_period('M')
    monthly_rent = df.groupby('lease_start_month')['monthly_rent'].mean().reset_index()
    monthly_rent['lease_start_month'] = monthly_rent['lease_start_month'].astype(str)
    
    fig_trend = px.line(
        monthly_rent, x='lease_start_month', y='monthly_rent',
        title="Average Monthly Rent Trend",
        labels={'lease_start_month': 'Month', 'monthly_rent': 'Average Rent ($)'}
    )
    fig_trend.update_xaxis(tickangle=45)
    st.plotly_chart(fig_trend, use_container_width=True)

def show_forecasting_tab():
    """Display forecasting dashboard."""
    st.header("🔮 Market Forecasting")
    
    df = get_data()
    
    st.subheader("Time Series Forecasting")
    
    # Forecasting parameters
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Parameters")
        forecast_days = st.slider("Forecast Days", 30, 365, 90)
        target_variable = st.selectbox(
            "Target Variable",
            ["monthly_rent", "payment_delays_last_6_months", "maintenance_requests_last_year"]
        )
        model_type = st.selectbox("Model Type", ["ARIMA", "Prophet", "Both"])
        
        if st.button("Run Forecast", type="primary"):
            with st.spinner("Training models and generating forecasts..."):
                try:
                    # Prepare time series data
                    ts_data = df.groupby('lease_start_date')[target_variable].mean().reset_index()
                    ts_data = ts_data.set_index('lease_start_date')[target_variable]
                    ts_data = ts_data.resample('D').mean().fillna(method='ffill')
                    
                    # Split data for forecasting
                    train_size = len(ts_data) - forecast_days
                    train_data = ts_data[:train_size]
                    test_data = ts_data[train_size:]
                    
                    # Store in session state
                    st.session_state.forecast_data = {
                        'train': train_data,
                        'test': test_data,
                        'full': ts_data,
                        'target': target_variable,
                        'forecast_days': forecast_days
                    }
                    
                    st.success("Forecast completed!")
                    
                except Exception as e:
                    st.error(f"Error during forecasting: {str(e)}")
    
    with col2:
        st.subheader("Forecast Results")
        
        if 'forecast_data' in st.session_state:
            forecast_data = st.session_state.forecast_data
            
            # Create forecast visualization
            fig = go.Figure()
            
            # Add historical data
            fig.add_trace(go.Scatter(
                x=forecast_data['train'].index,
                y=forecast_data['train'].values,
                mode='lines',
                name='Historical Data',
                line=dict(color='blue')
            ))
            
            # Add test data (if available)
            if len(forecast_data['test']) > 0:
                fig.add_trace(go.Scatter(
                    x=forecast_data['test'].index,
                    y=forecast_data['test'].values,
                    mode='lines',
                    name='Actual (Test)',
                    line=dict(color='green')
                ))
            
            # Simple forecast using linear trend (placeholder)
            last_date = forecast_data['train'].index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=forecast_data['forecast_days'],
                freq='D'
            )
            
            # Simple linear forecast
            recent_values = forecast_data['train'].tail(30).values
            trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
            last_value = forecast_data['train'].iloc[-1]
            
            forecast_values = [last_value + trend * i for i in range(1, forecast_data['forecast_days'] + 1)]
            
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_values,
                mode='lines',
                name='Forecast',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title=f"Forecast for {forecast_data['target'].replace('_', ' ').title()}",
                xaxis_title="Date",
                yaxis_title=forecast_data['target'].replace('_', ' ').title(),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                current_avg = forecast_data['train'].tail(30).mean()
                st.metric("Current Avg (30d)", f"{current_avg:.2f}")
            
            with col2:
                forecast_avg = np.mean(forecast_values)
                st.metric("Forecast Avg", f"{forecast_avg:.2f}")
            
            with col3:
                change = ((forecast_avg - current_avg) / current_avg) * 100
                st.metric("Predicted Change", f"{change:+.1f}%")
        
        else:
            st.info("Configure parameters and click 'Run Forecast' to see results.")

def show_data_quality_tab():
    """Display data quality dashboard."""
    st.header("🔍 Data Quality Analysis")
    
    df = get_data()
    
    # Generate quality report
    quality_report = DataQualityReport(df)
    report = quality_report.generate_report()
    
    # Overview metrics
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows", f"{report['overview']['shape'][0]:,}")
    
    with col2:
        st.metric("Columns", f"{report['overview']['shape'][1]:,}")
    
    with col3:
        memory_mb = report['overview']['memory_usage'] / (1024 * 1024)
        st.metric("Memory Usage", f"{memory_mb:.1f} MB")
    
    with col4:
        total_duplicates = report['duplicates']['total_duplicates']
        st.metric("Duplicates", f"{total_duplicates:,}")
    
    # Missing data analysis
    st.subheader("Missing Data Analysis")
    missing_data = pd.DataFrame([
        {"Column": col, "Missing Count": stats['count'], "Missing %": stats['percentage']}
        for col, stats in report['missing_data'].items()
        if stats['count'] > 0
    ])
    
    if not missing_data.empty:
        fig_missing = px.bar(
            missing_data, x='Column', y='Missing %',
            title="Missing Data by Column",
            color='Missing %',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_missing, use_container_width=True)
    else:
        st.success("No missing data found!")
    
    # Outlier analysis
    st.subheader("Outlier Analysis")
    outlier_data = pd.DataFrame([
        {"Column": col, "Outlier Count": stats['count'], "Outlier %": stats['percentage']}
        for col, stats in report['outliers'].items()
        if stats['count'] > 0
    ])
    
    if not outlier_data.empty:
        fig_outliers = px.bar(
            outlier_data, x='Column', y='Outlier %',
            title="Outliers by Column",
            color='Outlier %',
            color_continuous_scale='Oranges'
        )
        st.plotly_chart(fig_outliers, use_container_width=True)
    else:
        st.success("No significant outliers detected!")
    
    # Correlation analysis
    st.subheader("Correlation Analysis")
    numeric_df = df.select_dtypes(include=[np.number])
    
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            title="Correlation Matrix",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # High correlations
        high_corrs = report['correlations'].get('high_correlations', [])
        if high_corrs:
            st.warning("High correlations detected:")
            for corr in high_corrs:
                st.write(f"• {corr['column1']} ↔ {corr['column2']}: {corr['correlation']}")

def show_model_performance_tab():
    """Display model performance dashboard."""
    st.header("🎯 Model Performance")
    
    # Mock model performance data
    models_performance = {
        'Model': ['Random Forest', 'Logistic Regression', 'Neural Network', 'ARIMA', 'Prophet'],
        'Accuracy': [0.87, 0.82, 0.89, np.nan, np.nan],
        'F1-Score': [0.85, 0.80, 0.88, np.nan, np.nan],
        'RMSE': [np.nan, np.nan, np.nan, 145.2, 132.8],
        'MAE': [np.nan, np.nan, np.nan, 98.5, 89.2],
        'Model Type': ['Classification', 'Classification', 'Classification', 'Time Series', 'Time Series']
    }
    
    performance_df = pd.DataFrame(models_performance)
    
    # Classification models
    st.subheader("Classification Models (Tenant Churn)")
    class_models = performance_df[performance_df['Model Type'] == 'Classification']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_acc = px.bar(
            class_models, x='Model', y='Accuracy',
            title="Model Accuracy Comparison",
            color='Accuracy',
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig_acc, use_container_width=True)
    
    with col2:
        fig_f1 = px.bar(
            class_models, x='Model', y='F1-Score',
            title="F1-Score Comparison",
            color='F1-Score',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_f1, use_container_width=True)
    
    # Time series models
    st.subheader("Time Series Models (Rent Forecasting)")
    ts_models = performance_df[performance_df['Model Type'] == 'Time Series']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_rmse = px.bar(
            ts_models, x='Model', y='RMSE',
            title="RMSE Comparison (Lower is Better)",
            color='RMSE',
            color_continuous_scale='Reds_r'
        )
        st.plotly_chart(fig_rmse, use_container_width=True)
    
    with col2:
        fig_mae = px.bar(
            ts_models, x='Model', y='MAE',
            title="MAE Comparison (Lower is Better)",
            color='MAE',
            color_continuous_scale='Oranges_r'
        )
        st.plotly_chart(fig_mae, use_container_width=True)
    
    # Model comparison table
    st.subheader("Model Performance Summary")
    st.dataframe(performance_df, use_container_width=True)

def main():
    """Main dashboard application."""
    st.title("🏘️ Market Trend Forecasting Dashboard")
    st.markdown("Comprehensive analytics and forecasting for real estate market trends")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    tab = st.sidebar.radio(
        "Select View",
        ["📊 Overview", "🔮 Forecasting", "🔍 Data Quality", "🎯 Model Performance"]
    )
    
    # Configuration sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Configuration")
    
    # Data refresh
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Download data
    if st.sidebar.button("📥 Download Sample Data"):
        df = get_data()
        csv = df.to_csv(index=False)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name="market_data.csv",
            mime="text/csv"
        )
    
    # Display selected tab
    if tab == "📊 Overview":
        show_overview_tab()
    elif tab == "🔮 Forecasting":
        show_forecasting_tab()
    elif tab == "🔍 Data Quality":
        show_data_quality_tab()
    elif tab == "🎯 Model Performance":
        show_model_performance_tab()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666666;'>"
        "Market Trend Forecasting Dashboard | Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()