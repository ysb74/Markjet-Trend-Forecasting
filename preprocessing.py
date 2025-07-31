import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import zscore
from datetime import datetime

def perform_data_cleaning(df):
    """
    Cleans the DataFrame by handling missing values, converting data types,
    and creating derived features.
    """
    print("--- Performing Data Cleaning and Type Conversion ---")
    date_cols = ['lease_start_date', 'lease_end_date', 'last_sold_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    numeric_cols = [
        'monthly_rent', 'payment_delays_last_6_months',
        'maintenance_requests_last_year', 'feedback_score_avg',
        'beds', 'baths', 'sqft', 'price'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in numeric_cols:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Filled missing values in '{col}' with median: {median_val}")

    df.dropna(subset=['lease_start_date', 'lease_end_date'], inplace=True)
    print("Dropped rows with missing critical date information.")

    df['lease_duration_days'] = (df['lease_end_date'] - df['lease_start_date']).dt.days
    df['days_since_last_sold'] = (pd.to_datetime('now') - df['last_sold_date']).dt.days.fillna(0)

    print("Data cleaning and type conversion complete.")
    return df

def handle_outliers(df, columns, strategy='cap'):
    """
    Handles outliers in specified numerical columns using the IQR method.
    """
    print(f"\n--- Handling Outliers ({strategy} strategy) ---")
    df_cleaned = df.copy()
    outlier_detected = False

    for col in columns:
        if col in df_cleaned.columns and pd.api.types.is_numeric_dtype(df_cleaned[col]):
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            num_outliers = ((df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)).sum()
            if num_outliers > 0:
                outlier_detected = True
                print(f"  Outliers detected in '{col}': {num_outliers} values outside [{lower_bound:.2f}, {upper_bound:.2f}]")

                if strategy == 'cap':
                    df_cleaned[col] = np.where(df_cleaned[col] < lower_bound, lower_bound, df_cleaned[col])
                    df_cleaned[col] = np.where(df_cleaned[col] > upper_bound, upper_bound, df_cleaned[col])
                    print(f"  Outliers in '{col}' capped to bounds.")
                elif strategy == 'remove':
                    initial_rows = len(df_cleaned)
                    df_cleaned = df_cleaned[~((df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound))]
                    print(f"  Removed {initial_rows - len(df_cleaned)} rows with outliers in '{col}'.")
    
    if not outlier_detected:
        print("No significant outliers detected in specified columns using IQR method.")
    
    return df_cleaned

def drop_redundant_values_and_columns(df):
    """
    Identifies and drops redundant columns (e.g., duplicates, highly correlated)
    or values.
    """
    print("\n--- Dropping Redundant Values/Columns ---")
    initial_cols = df.columns.tolist()
    cols_to_drop_single_value = [col for col in df.columns if df[col].nunique() <= 1 and col not in ['lease_renewal', 'failure_imminent']]
    if cols_to_drop_single_value:
        df.drop(columns=cols_to_drop_single_value, inplace=True)
        print(f"Dropped columns with single unique value: {cols_to_drop_single_value}")

    transpose_df = df.T
    duplicated_cols = transpose_df[transpose_df.duplicated()].index.tolist()
    if duplicated_cols:
        df.drop(columns=duplicated_cols, inplace=True)
        print(f"Dropped exact duplicate columns: {duplicated_cols}")

    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop_high_corr = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        
        if to_drop_high_corr:
            print(f"Highly correlated columns (consider dropping one from each pair): {to_drop_high_corr}")
        else:
            print("No highly correlated numerical columns (above 0.95) found to drop automatically.")
    
    print("Redundancy check complete.")
    return df

def statistical_analysis_summary(df):
    """
    Provides a comprehensive statistical summary of the DataFrame,
    including various visualizations for better data understanding.
    """
    print("\n--- Statistical Analysis Summary ---")
    print("\nDataFrame Info:")
    df.info()
    print("\nDescriptive Statistics for Numerical Columns:")
    print(df.describe().T)
    print("\nValue Counts for Key Categorical/Discrete Columns:")
    categorical_cols = ['property_id', 'lease_renewal', 'beds', 'baths']
    for col in categorical_cols:
        if col in df.columns:
            print(f"\n--- {col} Distribution ---")
            print(df[col].value_counts(normalize=True) * 100)

    print("\nCorrelation Matrix (Numerical Features):")
    numerical_df = df.select_dtypes(include=np.number)
    if not numerical_df.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlation Matrix of Numerical Features')
        plt.show()
    else:
        print("No numerical columns for correlation analysis.")

    print("\n--- Visualizing Feature Distributions ---")
    numerical_features_to_plot = [
        'monthly_rent', 'payment_delays_last_6_months',
        'maintenance_requests_last_year', 'feedback_score_avg',
        'sqft', 'price', 'lease_duration_days', 'days_since_last_sold'
    ]
    
    num_plots = len(numerical_features_to_plot)
    if num_plots > 0:
        fig, axes = plt.subplots(nrows=int(np.ceil(num_plots/3)), ncols=3, figsize=(18, 5 * int(np.ceil(num_plots/3))))
        axes = axes.flatten()
        
        for i, col in enumerate(numerical_features_to_plot):
            if col in df.columns:
                sns.histplot(df[col], kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
            else:
                axes[i].set_visible(False)
        plt.tight_layout()
        plt.show()
    else:
        print("No numerical features available for distribution plots.")

    categorical_features_to_plot = ['lease_renewal', 'beds', 'baths']
    num_plots_cat = len(categorical_features_to_plot)
    if num_plots_cat > 0:
        fig_cat, axes_cat = plt.subplots(nrows=1, ncols=num_plots_cat, figsize=(6 * num_plots_cat, 6))
        if num_plots_cat == 1:
            axes_cat = [axes_cat]
        
        for i, col in enumerate(categorical_features_to_plot):
            if col in df.columns:
                sns.countplot(x=col, data=df, ax=axes_cat[i], palette='viridis')
                axes_cat[i].set_title(f'Count of {col}')
                axes_cat[i].set_xlabel(col)
                axes_cat[i].set_ylabel('Count')
                if col == 'lease_renewal':
                    axes_cat[i].set_xticklabels(['Churn (0)', 'Renewal (1)'])
            else:
                axes_cat[i].set_visible(False)
        plt.tight_layout()
        plt.show()
    else:
        print("No categorical features available for count plots.")

    print("\n--- Visualizing Relationships Between Key Features ---")
    
    if 'sqft' in df.columns and 'price' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='sqft', y='price', data=df, alpha=0.6)
        plt.title('Property Price vs. Square Footage')
        plt.xlabel('Square Footage')
        plt.ylabel('Price')
        plt.grid(True)
        plt.show()
    else:
        print("Cannot plot Price vs. Square Footage: 'sqft' or 'price' column missing.")

    if 'monthly_rent' in df.columns and 'sqft' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='sqft', y='monthly_rent', data=df, alpha=0.6)
        plt.title('Monthly Rent vs. Square Footage')
        plt.xlabel('Square Footage')
        plt.ylabel('Monthly Rent')
        plt.grid(True)
        plt.legend()
        plt.show()
    else:
        print("Cannot plot Monthly Rent vs. Square Footage: 'monthly_rent' or 'sqft' column missing.")

    if 'monthly_rent' in df.columns and 'lease_renewal' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='lease_renewal', y='monthly_rent', data=df, palette='pastel')
        plt.title('Monthly Rent Distribution by Lease Renewal Status')
        plt.xlabel('Lease Renewal (0: Churn, 1: Renew)')
        plt.ylabel('Monthly Rent')
        plt.show()
    else:
        print("Cannot plot Monthly Rent by Lease Renewal: 'monthly_rent' or 'lease_renewal' column missing.")

def customer_behavior_summary(df):
    """
    Provides a summary of customer (tenant) behavior metrics.
    """
    print("\n--- Customer Behavior Summary ---")
    print("Columns in DataFrame for customer_behavior_summary:", df.columns.tolist())

    if 'lease_renewal' in df.columns:
        churn_rate = (df['lease_renewal'] == 0).mean() * 100
        print(f"Overall Tenant Churn Rate: {churn_rate:.2f}%")

    required_cols_mean = ['payment_delays_last_6_months', 'maintenance_requests_last_year', 'feedback_score_avg']
    present_cols_mean = [col for col in required_cols_mean if col in df.columns]

    if present_cols_mean:
        print("\nAverage Payment Delays and Maintenance Requests:")
        print(df[present_cols_mean].mean())
    else:
        print("\nCould not calculate average payment delays and maintenance requests: Required columns missing.")

    print("\nCustomer Behavior by Lease Renewal Status:")
    if 'lease_renewal' in df.columns:
        group_cols = ['payment_delays_last_6_months', 'maintenance_requests_last_year', 'feedback_score_avg', 'monthly_rent']
        present_group_cols = [col for col in group_cols if col in df.columns]
        if present_group_cols:
            print(df.groupby('lease_renewal')[present_group_cols].mean())
        else:
            print("Could not group by lease renewal status: Required columns missing.")
    else:
        print("Cannot summarize by lease renewal status: 'lease_renewal' column missing.")

    print("\nTop Properties by Maintenance Requests:")
    if 'property_id' in df.columns and 'maintenance_requests_last_year' in df.columns:
        prop_maintenance = df.groupby('property_id')['maintenance_requests_last_year'].sum().sort_values(ascending=False)
        print(prop_maintenance.head())
    else:
        print("Cannot summarize top properties by maintenance requests: 'property_id' or 'maintenance_requests_last_year' column missing.")

    print("\nDistribution of Feedback Scores:")
    if 'feedback_score_avg' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df['feedback_score_avg'], bins=10, kde=True)
        plt.title('Distribution of Average Tenant Feedback Scores')
        plt.xlabel('Feedback Score')
        plt.ylabel('Number of Tenants')
        plt.show()
    else:
        print("Cannot plot feedback score distribution: 'feedback_score_avg' column missing.")

def prepare_for_modeling(df, target_column_classification='lease_renewal', target_column_regression='monthly_rent'):
    """
    Prepares the DataFrame for machine learning modeling.
    Returns:
        tuple: (X_classification, y_classification, X_regression, y_regression, preprocessed_df, scaler_classification, scaler_regression)
    """
    print("\n--- Preparing Data for Modeling and Visualization ---")
    df_model_ready = df.copy()

    date_cols_to_drop = ['lease_start_date', 'lease_end_date', 'last_sold_date']
    df_model_ready.drop(columns=[col for col in date_cols_to_drop if col in df_model_ready.columns], inplace=True)
    print(f"Dropped original date columns: {date_cols_to_drop}")

    if 'property_id' in df_model_ready.columns:
        le = LabelEncoder()
        df_model_ready['property_id_encoded'] = le.fit_transform(df_model_ready['property_id'])
        print(f"Encoded 'property_id' to numerical: {le.classes_} -> {le.transform(le.classes_)}")
        df_model_ready.drop(columns=['property_id'], inplace=True)

    if 'tenant_id' in df_model_ready.columns:
        df_model_ready.drop(columns=['tenant_id'], inplace=True)
        print("Dropped 'tenant_id' column.")

    X_classification = None
    y_classification = None
    X_regression = None
    y_regression = None

    if target_column_classification in df_model_ready.columns:
        y_classification = df_model_ready[target_column_classification]
        X_classification = df_model_ready.drop(columns=[target_column_classification])
        if target_column_regression in X_classification.columns:
            X_classification = X_classification.drop(columns=[target_column_regression])
        print(f"Separated classification target '{target_column_classification}' from features.")
    else:
        print(f"Warning: Classification target column '{target_column_classification}' not found. X_classification will be df_model_ready without regression target.")
        X_classification = df_model_ready.drop(columns=[target_column_regression]) if target_column_regression in df_model_ready.columns else df_model_ready

    if target_column_regression in df_model_ready.columns:
        y_regression = df_model_ready[target_column_regression]
        X_regression = df_model_ready.drop(columns=[target_column_regression])
        if target_column_classification in X_regression.columns:
            X_regression = X_regression.drop(columns=[target_column_classification])
        print(f"Separated regression target '{target_column_regression}' from features.")
    else:
        print(f"Warning: Regression target column '{target_column_regression}' not found. X_regression will be df_model_ready without classification target.")
        X_regression = df_model_ready.drop(columns=[target_column_classification]) if target_column_classification in df_model_ready.columns else df_model_ready

    numerical_cols_classification = X_classification.select_dtypes(include=np.number).columns.tolist() if X_classification is not None else []
    numerical_cols_regression = X_regression.select_dtypes(include=np.number).columns.tolist() if X_regression is not None else []

    scaler_classification = StandardScaler()
    scaler_regression = StandardScaler()

    if X_classification is not None and not X_classification.empty and numerical_cols_classification:
        X_classification_scaled = X_classification.copy()
        X_classification_scaled[numerical_cols_classification] = scaler_classification.fit_transform(X_classification[numerical_cols_classification])
        X_classification = X_classification_scaled
        print("Numerical features for classification scaled.")
    else:
        print("No numerical features to scale for classification or X_classification is empty.")
        scaler_classification = None

    if X_regression is not None and not X_regression.empty and numerical_cols_regression:
        X_regression_scaled = X_regression.copy()
        X_regression_scaled[numerical_cols_regression] = scaler_regression.fit_transform(X_regression[numerical_cols_regression])
        X_regression = X_regression_scaled
        print("Numerical features for regression scaled.")
    else:
        print("No numerical features to scale for regression or X_regression is empty.")
        scaler_regression = None
    
    print(f"Features (X_classification) shape: {X_classification.shape if X_classification is not None else 'N/A'}")
    print(f"Target (y_classification) shape: {y_classification.shape if y_classification is not None else 'N/A'}")
    print(f"Features (X_regression) shape: {X_regression.shape if X_regression is not None else 'N/A'}")
    print(f"Target (y_regression) shape: {y_regression.shape if y_regression is not None else 'N/A'}")
    print("Data ready for modeling.")
    return X_classification, y_classification, X_regression, y_regression, df_model_ready, scaler_classification, scaler_regression