# Market Trend Forecasting - Model Evaluation
# This notebook focuses on model training, evaluation, and comparison

# %% [markdown]
# # Model Evaluation and Comparison
# 
# This notebook focuses on training, evaluating, and comparing different machine learning models for market trend forecasting.
# 
# ## Table of Contents
# 1. [Data Preparation](#data-preparation)
# 2. [Model Training](#model-training)
# 3. [Model Evaluation](#model-evaluation)
# 4. [Model Comparison](#model-comparison)
# 5. [Hyperparameter Tuning](#hyperparameter-tuning)
# 6. [Feature Importance Analysis](#feature-importance)
# 7. [Model Deployment Preparation](#deployment)

# %% [code]
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
import sys
sys.path.append('../')

from src.data.ingestion import load_crm_data_live
from src.data.preprocessing import perform_data_cleaning, prepare_for_modeling
from src.models.ml_models import train_evaluate_classification_model, train_evaluate_regression_model
from src.utils.logger import get_logger

logger = get_logger(__name__)

print("🤖 Model Evaluation libraries imported successfully!")

# %% [markdown]
# ## 1. Data Preparation {#data-preparation}

# %% [code]
# Load and prepare data
df = load_crm_data_live()
df_cleaned = perform_data_cleaning(df.copy())

print("📊 Dataset loaded and cleaned!")
print(f"Original shape: {df.shape}")
print(f"Cleaned shape: {df_cleaned.shape}")

# %% [code]
# Prepare data for modeling
X_classification, y_classification, X_regression, y_regression, df_for_viz, scaler_classification, scaler_regression = \
    prepare_for_modeling(df_cleaned.copy(), target_column_classification='lease_renewal', target_column_regression='monthly_rent')

print("🎯 Data prepared for modeling:")
print(f"Classification features shape: {X_classification.shape if X_classification is not None else 'None'}")
print(f"Classification target shape: {y_classification.shape if y_classification is not None else 'None'}")
print(f"Regression features shape: {X_regression.shape if X_regression is not None else 'None'}")
print(f"Regression target shape: {y_regression.shape if y_regression is not None else 'None'}")

# %% [markdown]
# ## 2. Model Training {#model-training}

# %% [code]
# Define models for classification
classification_models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(random_state=42, probability=True),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# Define models for regression
regression_models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting Regressor': GradientBoostingClassifier(random_state=42)
}

print("📋 Models defined for training!")

# %% [code]
# Train classification models
classification_results = {}

if X_classification is not None and y_classification is not None:
    print("🎯 Training Classification Models...")
    
    for name, model in classification_models.items():
        print(f"\n--- Training {name} ---")
        
        # Train and evaluate using custom function
        result = train_evaluate_classification_model(X_classification, y_classification, model, name)
        classification_results[name] = result
        
        if result is not None:
            print(f"✅ {name} training completed!")
        else:
            print(f"❌ {name} training failed!")

# %% [code]
# Train regression models
regression_results = {}

if X_regression is not None and y_regression is not None:
    print("\n🎯 Training Regression Models...")
    
    for name, model in regression_models.items():
        print(f"\n--- Training {name} ---")
        
        # Train and evaluate using custom function
        result = train_evaluate_regression_model(X_regression, y_regression, model, name, "Monthly Rent")
        regression_results[name] = result
        
        if result is not None:
            print(f"✅ {name} training completed!")
        else:
            print(f"❌ {name} training failed!")

# %% [markdown]
# ## 3. Model Evaluation {#model-evaluation}

# %% [code]
# Evaluate classification models
if classification_results:
    print("📊 Classification Model Evaluation Results:")
    print("=" * 60)
    
    # Create comparison DataFrame
    comparison_data = []
    
    for name, result in classification_results.items():
        if result is not None:
            comparison_data.append({
                'Model': name,
                'Accuracy': result.get('accuracy', 0),
                'Precision': result.get('precision', 0),
                'Recall': result.get('recall', 0),
                'F1-Score': result.get('f1_score', 0),
                'ROC-AUC': result.get('roc_auc', 0)
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.round(4))
        
        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        for i, metric in enumerate(metrics):
            if i < len(axes):
                comparison_df.plot(x='Model', y=metric, kind='bar', ax=axes[i])
                axes[i].set_title(f'{metric} Comparison')
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

# %% [code]
# Evaluate regression models
if regression_results:
    print("\n📊 Regression Model Evaluation Results:")
    print("=" * 60)
    
    # Create comparison DataFrame
    comparison_data = []
    
    for name, result in regression_results.items():
        if result is not None:
            comparison_data.append({
                'Model': name,
                'R² Score': result.get('r2_score', 0),
                'MSE': result.get('mse', 0),
                'RMSE': result.get('rmse', 0),
                'MAE': result.get('mae', 0)
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.round(4))
        
        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        metrics = ['R² Score', 'MSE', 'RMSE', 'MAE']
        for i, metric in enumerate(metrics):
            if i < len(axes):
                comparison_df.plot(x='Model', y=metric, kind='bar', ax=axes[i])
                axes[i].set_title(f'{metric} Comparison')
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ## 4. Model Comparison {#model-comparison}

# %% [code]
# Cross-validation comparison
if X_classification is not None and y_classification is not None:
    print("🔄 Cross-Validation Results (Classification):")
    print("=" * 50)
    
    cv_results = {}
    
    for name, model in classification_models.items():
        try:
            cv_scores = cross_val_score(model, X_classification, y_classification, cv=5, scoring='accuracy')
            cv_results[name] = {
                'mean_cv_score': cv_scores.mean(),
                'std_cv_score': cv_scores.std(),
                'cv_scores': cv_scores
            }
            print(f"{name}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        except Exception as e:
            print(f"{name}: Error - {e}")
    
    # Plot CV results
    if cv_results:
        cv_df = pd.DataFrame([
            {
                'Model': name,
                'Mean CV Score': results['mean_cv_score'],
                'Std CV Score': results['std_cv_score']
            }
            for name, results in cv_results.items()
        ])
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(cv_df['Model'], cv_df['Mean CV Score'], yerr=cv_df['Std CV Score'], capsize=5)
        plt.title('Cross-Validation Accuracy Scores')
        plt.ylabel('Accuracy')
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ## 5. Hyperparameter Tuning {#hyperparameter-tuning}

# %% [code]
# Hyperparameter tuning for best classification model
if X_classification is not None and y_classification is not None:
    print("🔧 Hyperparameter Tuning for Random Forest:")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Grid search
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_classification, y_classification)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Train best model
    best_rf = grid_search.best_estimator_
    best_result = train_evaluate_classification_model(X_classification, y_classification, best_rf, "Optimized Random Forest")
    
    if best_result is not None:
        print("✅ Optimized Random Forest training completed!")

# %% [markdown]
# ## 6. Feature Importance Analysis {#feature-importance}

# %% [code]
# Analyze feature importance for Random Forest
if X_classification is not None and y_classification is not None:
    print("🎯 Feature Importance Analysis:")
    
    # Train Random Forest for feature importance
    rf_importance = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_importance.fit(X_classification, y_classification)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X_classification.columns,
        'importance': rf_importance.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    sns.barplot(data=top_features, x='importance', y='feature')
    plt.title('Top 15 Feature Importance (Random Forest)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 7. Model Deployment Preparation {#deployment}

# %% [code]
# Save best models and results
import joblib
import json
from datetime import datetime

# Create deployment directory
import os
os.makedirs('../deployed_models', exist_ok=True)

# Save best classification model
if classification_results:
    best_classification_model = max(classification_results.items(), 
                                  key=lambda x: x[1]['accuracy'] if x[1] else 0)
    
    model_name = best_classification_model[0]
    model_result = best_classification_model[1]
    
    if model_result is not None:
        # Save model
        model_path = f'../deployed_models/best_classification_model.joblib'
        joblib.dump(model_result['model'], model_path)
        print(f"💾 Best classification model saved: {model_path}")
        
        # Save model metadata
        metadata = {
            'model_name': model_name,
            'model_type': 'classification',
            'accuracy': model_result['accuracy'],
            'precision': model_result['precision'],
            'recall': model_result['recall'],
            'f1_score': model_result['f1_score'],
            'roc_auc': model_result['roc_auc'],
            'deployment_date': datetime.now().isoformat(),
            'features': X_classification.columns.tolist() if X_classification is not None else []
        }
        
        metadata_path = f'../deployed_models/best_classification_model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"📋 Model metadata saved: {metadata_path}")

# Save best regression model
if regression_results:
    best_regression_model = max(regression_results.items(), 
                               key=lambda x: x[1]['r2_score'] if x[1] else 0)
    
    model_name = best_regression_model[0]
    model_result = best_regression_model[1]
    
    if model_result is not None:
        # Save model
        model_path = f'../deployed_models/best_regression_model.joblib'
        joblib.dump(model_result['model'], model_path)
        print(f"💾 Best regression model saved: {model_path}")
        
        # Save model metadata
        metadata = {
            'model_name': model_name,
            'model_type': 'regression',
            'r2_score': model_result['r2_score'],
            'mse': model_result['mse'],
            'rmse': model_result['rmse'],
            'mae': model_result['mae'],
            'deployment_date': datetime.now().isoformat(),
            'features': X_regression.columns.tolist() if X_regression is not None else []
        }
        
        metadata_path = f'../deployed_models/best_regression_model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"📋 Model metadata saved: {metadata_path}")

# Save scalers
if scaler_classification is not None:
    scaler_path = '../deployed_models/classification_scaler.joblib'
    joblib.dump(scaler_classification, scaler_path)
    print(f"🔧 Classification scaler saved: {scaler_path}")

if scaler_regression is not None:
    scaler_path = '../deployed_models/regression_scaler.joblib'
    joblib.dump(scaler_regression, scaler_path)
    print(f"🔧 Regression scaler saved: {scaler_path}")

# %% [code]
# Generate deployment report
print("\n📊 Model Evaluation Summary:")
print("=" * 50)

if classification_results:
    print("\n🏆 Best Classification Model:")
    best_classification = max(classification_results.items(), 
                            key=lambda x: x[1]['accuracy'] if x[1] else 0)
    print(f"Model: {best_classification[0]}")
    print(f"Accuracy: {best_classification[1]['accuracy']:.4f}")
    print(f"Precision: {best_classification[1]['precision']:.4f}")
    print(f"Recall: {best_classification[1]['recall']:.4f}")
    print(f"F1-Score: {best_classification[1]['f1_score']:.4f}")

if regression_results:
    print("\n🏆 Best Regression Model:")
    best_regression = max(regression_results.items(), 
                         key=lambda x: x[1]['r2_score'] if x[1] else 0)
    print(f"Model: {best_regression[0]}")
    print(f"R² Score: {best_regression[1]['r2_score']:.4f}")
    print(f"MSE: {best_regression[1]['mse']:.4f}")
    print(f"RMSE: {best_regression[1]['rmse']:.4f}")
    print(f"MAE: {best_regression[1]['mae']:.4f}")

print("\n✅ Model evaluation completed!")
print("\n📋 Next Steps:")
print("1. Review model performance metrics")
print("2. Deploy best models to production")
print("3. Set up model monitoring")
print("4. Create prediction API endpoints") 