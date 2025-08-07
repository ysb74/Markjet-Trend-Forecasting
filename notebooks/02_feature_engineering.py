# Market Trend Forecasting - Feature Engineering
# This notebook focuses on creating and engineering features for the ML models

# %% [markdown]
# # Feature Engineering for Market Trend Forecasting
# 
# This notebook focuses on creating and engineering features that will improve our machine learning models' performance.
# 
# ## Table of Contents
# 1. [Feature Creation](#feature-creation)
# 2. [Feature Selection](#feature-selection)
# 3. [Feature Scaling](#feature-scaling)
# 4. [Feature Importance Analysis](#feature-importance)
# 5. [Model Preparation](#model-preparation)

# %% [code]
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
import sys
sys.path.append('../')

from src.data.ingestion import load_crm_data_live
from src.data.preprocessing import perform_data_cleaning, handle_outliers
from src.utils.logger import get_logger

logger = get_logger(__name__)

print("🔧 Feature Engineering libraries imported successfully!")

# %% [markdown]
# ## 1. Feature Creation {#feature-creation}

# %% [code]
# Load and clean the data
df = load_crm_data_live()
df_cleaned = perform_data_cleaning(df.copy())

print("📊 Original dataset shape:", df.shape)
print("🧹 Cleaned dataset shape:", df_cleaned.shape)

# %% [code]
# Create time-based features
if 'lease_start_date' in df_cleaned.columns:
    df_cleaned['lease_start_date'] = pd.to_datetime(df_cleaned['lease_start_date'])
    
    # Extract temporal features
    df_cleaned['lease_start_year'] = df_cleaned['lease_start_date'].dt.year
    df_cleaned['lease_start_month'] = df_cleaned['lease_start_date'].dt.month
    df_cleaned['lease_start_quarter'] = df_cleaned['lease_start_date'].dt.quarter
    df_cleaned['lease_start_dayofweek'] = df_cleaned['lease_start_date'].dt.dayofweek
    df_cleaned['lease_start_season'] = df_cleaned['lease_start_date'].dt.month.map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })

if 'lease_end_date' in df_cleaned.columns:
    df_cleaned['lease_end_date'] = pd.to_datetime(df_cleaned['lease_end_date'])
    df_cleaned['lease_duration_days'] = (df_cleaned['lease_end_date'] - df_cleaned['lease_start_date']).dt.days
    df_cleaned['lease_duration_months'] = df_cleaned['lease_duration_days'] / 30.44

print("⏰ Time-based features created!")

# %% [code]
# Create risk and behavior features
if all(col in df_cleaned.columns for col in ['payment_delays_last_6_months', 'maintenance_requests_last_year']):
    # Risk score (weighted combination)
    df_cleaned['risk_score'] = (
        df_cleaned['payment_delays_last_6_months'] * 0.6 + 
        df_cleaned['maintenance_requests_last_year'] * 0.4
    )
    
    # Risk categories
    df_cleaned['risk_category'] = pd.cut(
        df_cleaned['risk_score'], 
        bins=[0, 1, 3, 10], 
        labels=['Low', 'Medium', 'High']
    )
    
    # Payment reliability
    df_cleaned['payment_reliability'] = 6 - df_cleaned['payment_delays_last_6_months']
    df_cleaned['payment_reliability'] = df_cleaned['payment_reliability'].clip(0, 6)
    
    # Maintenance frequency
    df_cleaned['maintenance_frequency'] = df_cleaned['maintenance_requests_last_year'] / 12

print("⚠️ Risk and behavior features created!")

# %% [code]
# Create financial features
if 'monthly_rent' in df_cleaned.columns:
    # Rent categories
    df_cleaned['rent_category'] = pd.cut(
        df_cleaned['monthly_rent'],
        bins=[0, 1200, 1800, 2500, float('inf')],
        labels=['Budget', 'Standard', 'Premium', 'Luxury']
    )
    
    # Rent per day (for lease duration analysis)
    if 'lease_duration_days' in df_cleaned.columns:
        df_cleaned['rent_per_day'] = df_cleaned['monthly_rent'] / 30.44
    
    # Annual rent
    df_cleaned['annual_rent'] = df_cleaned['monthly_rent'] * 12

print("💰 Financial features created!")

# %% [code]
# Create interaction features
if all(col in df_cleaned.columns for col in ['monthly_rent', 'payment_delays_last_6_months']):
    # Rent-delay interaction
    df_cleaned['rent_delay_ratio'] = df_cleaned['payment_delays_last_6_months'] / df_cleaned['monthly_rent']
    
    # Rent-maintenance interaction
    if 'maintenance_requests_last_year' in df_cleaned.columns:
        df_cleaned['rent_maintenance_ratio'] = df_cleaned['maintenance_requests_last_year'] / df_cleaned['monthly_rent']

print("🔄 Interaction features created!")

# %% [markdown]
# ## 2. Feature Selection {#feature-selection}

# %% [code]
# Prepare features for selection
# Select numerical features for analysis
numerical_features = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()

# Remove target variables if they exist
target_vars = ['lease_renewal']
for target in target_vars:
    if target in numerical_features:
        numerical_features.remove(target)

print(f"📈 Analyzing {len(numerical_features)} numerical features:")
print(numerical_features)

# %% [code]
# Feature importance using Random Forest
if 'lease_renewal' in df_cleaned.columns:
    # Prepare data
    X = df_cleaned[numerical_features].fillna(0)
    y = df_cleaned['lease_renewal']
    
    # Train Random Forest for feature importance
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X, y)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': numerical_features,
        'importance': rf_classifier.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("🎯 Feature Importance (Classification):")
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
    plt.title('Top 15 Feature Importance for Lease Renewal Prediction')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

# %% [code]
# Feature selection using statistical tests
if 'lease_renewal' in df_cleaned.columns:
    # Select top features using ANOVA F-test
    selector = SelectKBest(score_func=f_classif, k=10)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    feature_scores = selector.scores_[selector.get_support()]
    
    print("📊 Top 10 Features (ANOVA F-test):")
    for feature, score in zip(selected_features, feature_scores):
        print(f"{feature}: {score:.2f}")

# %% [markdown]
# ## 3. Feature Scaling {#feature-scaling}

# %% [code]
# Prepare features for scaling
features_to_scale = [
    'monthly_rent', 'payment_delays_last_6_months', 
    'maintenance_requests_last_year', 'feedback_score',
    'lease_duration_days', 'risk_score', 'payment_reliability',
    'maintenance_frequency', 'rent_per_day', 'annual_rent'
]

# Filter features that exist in the dataset
features_to_scale = [f for f in features_to_scale if f in df_cleaned.columns]

print(f"🔢 Scaling {len(features_to_scale)} features:")
print(features_to_scale)

# %% [code]
# Apply StandardScaler
scaler = StandardScaler()
df_scaled = df_cleaned.copy()

# Scale numerical features
df_scaled[features_to_scale] = scaler.fit_transform(df_cleaned[features_to_scale])

print("✅ Features scaled successfully!")

# %% [code]
# Verify scaling results
print("📊 Scaling verification:")
for feature in features_to_scale[:5]:  # Show first 5 features
    print(f"{feature}:")
    print(f"  Original - Mean: {df_cleaned[feature].mean():.2f}, Std: {df_cleaned[feature].std():.2f}")
    print(f"  Scaled   - Mean: {df_scaled[feature].mean():.2f}, Std: {df_scaled[feature].std():.2f}")

# %% [markdown]
# ## 4. Feature Importance Analysis {#feature-importance}

# %% [code]
# Correlation analysis with target
if 'lease_renewal' in df_scaled.columns:
    correlations = df_scaled[numerical_features + ['lease_renewal']].corr()['lease_renewal'].abs().sort_values(ascending=False)
    
    print("🔗 Feature Correlations with Lease Renewal:")
    print(correlations)
    
    # Plot top correlations
    plt.figure(figsize=(10, 8))
    correlations[1:11].plot(kind='bar')  # Exclude lease_renewal itself
    plt.title('Top 10 Feature Correlations with Lease Renewal')
    plt.xlabel('Features')
    plt.ylabel('Absolute Correlation')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# %% [code]
# Feature distribution by target
if 'lease_renewal' in df_scaled.columns:
    # Select top 6 features for visualization
    top_features = feature_importance.head(6)['feature'].tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, feature in enumerate(top_features):
        if i < len(axes):
            # Box plot by renewal status
            df_scaled.boxplot(column=feature, by='lease_renewal', ax=axes[i])
            axes[i].set_title(f'{feature} by Renewal Status')
            axes[i].set_xlabel('Lease Renewal (0=Churned, 1=Renewed)')
            axes[i].set_ylabel(feature)
    
    plt.suptitle('Feature Distributions by Lease Renewal Status')
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 5. Model Preparation {#model-preparation}

# %% [code]
# Prepare final feature set for modeling
# Select the most important features
top_features = feature_importance.head(15)['feature'].tolist()

# Add categorical features
categorical_features = ['risk_category', 'rent_category', 'lease_start_season']
categorical_features = [f for f in categorical_features if f in df_scaled.columns]

# Combine numerical and categorical features
final_features = top_features + categorical_features

print(f"🎯 Final feature set ({len(final_features)} features):")
print(final_features)

# %% [code]
# Encode categorical features
df_final = df_scaled.copy()

for feature in categorical_features:
    if feature in df_final.columns:
        le = LabelEncoder()
        df_final[f'{feature}_encoded'] = le.fit_transform(df_final[feature].astype(str))
        print(f"✅ Encoded {feature}")

# %% [code]
# Create final dataset for modeling
modeling_features = [f for f in final_features if f in df_final.columns]
modeling_features.extend([f'{f}_encoded' for f in categorical_features if f'{f}_encoded' in df_final.columns])

X_final = df_final[modeling_features].fillna(0)
y_final = df_final['lease_renewal'] if 'lease_renewal' in df_final.columns else None

print(f"📊 Final modeling dataset shape: {X_final.shape}")
print(f"🎯 Target variable available: {y_final is not None}")

# %% [code]
# Save processed data
output_path = '../data/processed/feature_engineered_data.csv'
df_final.to_csv(output_path, index=False)
print(f"💾 Feature engineered data saved to {output_path}")

# Save feature list
feature_list_path = '../data/processed/modeling_features.txt'
with open(feature_list_path, 'w') as f:
    f.write('\n'.join(modeling_features))
print(f"📝 Feature list saved to {feature_list_path}")

# Save scaler for later use
import joblib
scaler_path = '../deployed_models/feature_scaler.joblib'
joblib.dump(scaler, scaler_path)
print(f"🔧 Scaler saved to {scaler_path}")

print("\n✅ Feature Engineering completed!")
print("\n📋 Summary:")
print(f"- Created {len(df_final.columns) - len(df.columns)} new features")
print(f"- Selected {len(modeling_features)} features for modeling")
print(f"- Scaled {len(features_to_scale)} numerical features")
print(f"- Encoded {len(categorical_features)} categorical features") 