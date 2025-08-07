import pandas as pd
import numpy as np
import os
import joblib
import sys
import os.path


# Import MLflow for MLOps capabilities
try:
    import mlflow
    import mlflow.pyfunc # For loading models from MLflow
    import mlflow.sklearn
    import mlflow.tensorflow
    print("MLflow imported successfully in main.")
    # Set MLflow tracking URI (local 'mlruns' directory by default)
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    print(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")
except ImportError:
    print("MLflow not installed. MLOps features will be skipped in main.")
    mlflow = None

# Import TensorFlow/Keras if needed for loading Keras models
try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    tf = None


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Import functions from our custom modules
from src.data.ingestion import load_crm_data_live, scrape_public_listings, get_property_data_from_official_api, aggregate_real_estate_data
from src.data.preprocessing import perform_data_cleaning, handle_outliers, drop_redundant_values_and_columns, statistical_analysis_summary, customer_behavior_summary, prepare_for_modeling
from src.models.ml_models import train_evaluate_classification_model, train_evaluate_regression_model, train_evaluate_neural_network_classification, perform_hyperparameter_optimization
from sklearn.metrics import accuracy_score, mean_squared_error # For monitoring checks

# --- Model Deployment, Monitoring, and Iteration (Conceptual) ---

def save_model(model, model_type, model_name, scaler=None):
    """
    Saves a trained model and its associated scaler.
    Emphasizes MLflow Model Registry for real-world deployment.
    """
    model_dir = "deployed_models"
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n--- Saving Model '{model_name}' ---")

    if mlflow is not None:
        try:
            # This will log and register the model. If it's already logged from training,
            # this will create a new version or update if the model name exists.
            if model_type == 'sklearn':
                mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)
            elif model_type == 'keras':
                mlflow.tensorflow.log_model(model, "model", registered_model_name=model_name)
            
            print(f"Model '{model_name}' registered in MLflow Model Registry.")
        except Exception as e:
            print(f"Error registering model '{model_name}' with MLflow: {e}. Proceeding with local save.")
    else:
        print("MLflow not available. Skipping MLflow Model Registry operations.")

    # Always save locally as a fallback or for direct file access
    if model_type == 'sklearn':
        filename = os.path.join(model_dir, f"{model_name.replace(' ', '_')}.joblib")
        joblib.dump(model, filename)
        print(f"Scikit-learn model '{model_name}' saved locally to {filename}")
        if scaler:
            scaler_filename = os.path.join(model_dir, f"{model_name.replace(' ', '_')}_scaler.joblib")
            joblib.dump(scaler, scaler_filename)
            print(f"Scaler for '{model_name}' saved locally to {scaler_filename}")
    elif model_type == 'keras':
        if tf is None:
            print("TensorFlow not installed. Cannot save Keras model locally.")
            return
        filename = os.path.join(model_dir, f"{model_name.replace(' ', '_')}.h5")
        model.save(filename)
        print(f"Keras model '{model_name}' saved locally to {filename}")
        if scaler:
            scaler_filename = os.path.join(model_dir, f"{model_name.replace(' ', '_')}_scaler.joblib")
            joblib.dump(scaler, scaler_filename)
            print(f"Scaler for '{model_name}' saved locally to {scaler_filename}")
    else:
        print(f"Unsupported model_type: {model_type}. Model not saved locally.")

def load_model_and_predict(model_uri, scaler_path=None, new_data=None):
    """
    Simulates loading a deployed model (preferably from MLflow Model Registry)
    and making predictions on new data.
    """
    print(f"\n--- Simulating Model Deployment and Prediction using URI: {model_uri} ---")
    if new_data is None or new_data.empty:
        print("No new data provided for prediction.")
        return np.array([])

    model = None
    scaler = None

    try:
        if mlflow is not None and model_uri.startswith("models:/"):
            model = mlflow.pyfunc.load_model(model_uri)
            print(f"Model loaded from MLflow Model Registry: {model_uri}")
        else:
            if model_uri.endswith(".joblib"):
                model = joblib.load(model_uri)
                print(f"Scikit-learn model loaded from local path: {model_uri}")
            elif model_uri.endswith(".h5") and tf is not None:
                model = keras.models.load_model(model_uri)
                print(f"Keras model loaded from local path: {model_uri}")
            else:
                print(f"Unsupported model URI or type for local loading: {model_uri}. Cannot load model.")
                return np.array([])

        if scaler_path and os.path.exists(scaler_path): # Check if scaler_path exists
            scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from {scaler_path}")
            numerical_cols_to_scale = new_data.select_dtypes(include=np.number).columns.tolist()
            if numerical_cols_to_scale:
                new_data_scaled = new_data.copy()
                new_data_scaled[numerical_cols_to_scale] = scaler.transform(new_data[numerical_cols_to_scale])
                new_data = new_data_scaled
                print("New data scaled using loaded scaler.")
            else:
                print("No numerical columns to scale in new data.")
        elif scaler_path:
            print(f"Warning: Scaler file not found at {scaler_path}. Proceeding without scaling new data.")


        predictions = model.predict(new_data)
        if isinstance(model, keras.Sequential) and predictions.shape[1] == 1:
            predictions = (predictions > 0.5).astype(int)
        print("Predictions made successfully.")
        return predictions

    except FileNotFoundError:
        print(f"Error: Model or scaler file not found at {model_uri} or {scaler_path}.")
    except Exception as e:
        print(f"An error occurred during model loading or prediction: {e}")
    return np.array([])

def simulate_monitoring(trained_model, model_name, X_train_original, y_train_original, scaler_fitted, prediction_type='classification'):
    """
    Simulates continuous monitoring of a deployed model.
    Highlights how MLflow's logged metrics can be used for baseline comparison.
    """
    print(f"\n--- Simulating Monitoring for {model_name} ---")
    print("Monitoring for data drift and performance degradation...")

    num_new_samples = 5
    simulated_new_data_features = pd.DataFrame(
        np.random.normal(X_train_original.mean(), X_train_original.std() * 1.05,
                         size=(num_new_samples, X_train_original.shape[1])),
        columns=X_train_original.columns
    )
    simulated_new_data_actuals = pd.Series(np.random.randint(0, 2, size=num_new_samples)) if prediction_type == 'classification' else \
                                 pd.Series(np.random.normal(y_train_original.mean(), y_train_original.std() * 1.05, size=num_new_samples))


    numerical_cols_to_scale = simulated_new_data_features.select_dtypes(include=np.number).columns.tolist()
    if scaler_fitted and numerical_cols_to_scale:
        simulated_new_data_features_scaled = simulated_new_data_features.copy()
        simulated_new_data_features_scaled[numerical_cols_to_scale] = scaler_fitted.transform(simulated_new_data_features[numerical_cols_to_scale])
        X_new_processed = simulated_new_data_features_scaled
    else:
        X_new_processed = simulated_new_data_features
    
    if X_new_processed.empty:
        print("No processed new data for monitoring.")
        return

    print("\n  -- Data Drift Check (Conceptual) --")
    drift_detected = False
    for col in X_train_original.columns:
        if col in X_new_processed.columns and pd.api.types.is_numeric_dtype(X_train_original[col]):
            train_mean = X_train_original[col].mean()
            new_mean = X_new_processed[col].mean()
            if abs(train_mean - new_mean) / train_mean > 0.1:
                print(f"    ALERT: Data drift detected in '{col}'! Train mean: {train_mean:.2f}, New mean: {new_mean:.2f}")
                drift_detected = True
    if not drift_detected:
        print("    No significant data drift detected in key numerical features (conceptual check).")
    print("    Real-world data drift monitoring involves more sophisticated statistical tests and dedicated tools.")

    print("\n  -- Model Performance Check (Conceptual) --")
    try:
        if prediction_type == 'classification':
            y_pred_new = trained_model.predict(X_new_processed)
            if hasattr(trained_model, 'predict_proba') and not isinstance(trained_model, keras.Sequential):
                y_pred_new_proba = trained_model.predict_proba(X_new_processed)[:, 1]
                y_pred_new = (y_pred_new_proba > 0.5).astype(int)
            elif isinstance(trained_model, keras.Sequential):
                y_pred_new_proba = trained_model.predict(X_new_processed, verbose=0).flatten()
                y_pred_new = (y_pred_new_proba > 0.5).astype(int)

            current_accuracy = accuracy_score(simulated_new_data_actuals, y_pred_new)
            baseline_accuracy = 0.8 # Placeholder: replace with actual test accuracy from MLflow run
            print(f"    Baseline Accuracy (from training/MLflow): {baseline_accuracy:.4f}")
            if current_accuracy < baseline_accuracy * 0.9:
                print(f"    ALERT: Model performance degraded! Current Accuracy: {current_accuracy:.4f}, Baseline: {baseline_accuracy:.4f}")
            else:
                print(f"    Model performance stable. Current Accuracy: {current_accuracy:.4f}")
        elif prediction_type == 'regression':
            y_pred_new = trained_model.predict(X_new_processed)
            current_mse = mean_squared_error(simulated_new_data_actuals, y_pred_new)
            baseline_mse = 100000 # Placeholder: replace with actual test MSE from MLflow run
            print(f"    Baseline MSE (from training/MLflow): {baseline_mse:.2f}")
            if current_mse > baseline_mse * 1.1:
                print(f"    ALERT: Model performance degraded! Current MSE: {current_mse:.2f}, Baseline: {baseline_mse:.2f}")
            else:
                print(f"    Model performance stable. Current MSE: {current_mse:.2f}")
    except Exception as e:
        print(f"    Error during model performance check: {e}")

    print("\nMonitoring cycle complete.")
    print("    Real-time monitoring systems (e.g., Prometheus, Grafana, cloud-native monitoring) would ingest these metrics and provide dashboards and alerts.")


def conceptual_iteration_loop():
    """
    Explains the conceptual iteration and continuous improvement loop.
    """
    print("\n--- Iteration and Continuous Improvement (Conceptual) ---")
    print("Insights from monitoring (e.g., detected data drift, performance degradation) trigger the iteration loop:")
    print("1. **Root Cause Analysis:** Investigate why drift or degradation occurred (e.g., changes in tenant demographics, market shifts, new property types).")
    print("2. **Data Re-collection/Augmentation:** Collect more recent data or new types of data that address the identified issues.")
    print("3. **Data Re-preprocessing:** Apply the cleaning, outlier handling, and feature engineering steps to the new/updated dataset.")
    print("4. **Model Retraining/Refinement:** Retrain existing models on the fresh data, or explore new model architectures/algorithms.")
    print("5. **Model Versioning & Promotion (via MLflow Model Registry):** Register the new, improved model as a new version in MLflow. Promote it through stages (Staging -> Production) after successful validation.")
    print("6. **Re-deployment:** Deploy the new model version, potentially using A/B testing or canary deployments.")
    print("7. **Repeat:** The monitoring process continues, starting a new cycle as needed.")
    print("\nThis iterative process ensures the ML models remain relevant and performant over time in a dynamic real estate market.")


if __name__ == "__main__":
    print("--- Starting Combined Real Estate Data Pipeline ---")

    crm_data = load_crm_data_live()
    print("\n--- CRM Data Loaded (Conceptual Live) ---")
    print(crm_data.head())

    scraped_data = scrape_public_listings(search_url='http://quotes.toscrape.com/', num_pages=1)
    print("\n--- Scraped Data Loaded (Conceptual) ---")
    if scraped_data:
        print(pd.DataFrame(scraped_data).head())
    else:
        print("No scraped data to display.")

    zillow_api_data = get_property_data_from_official_api(api_name='Zillow', location='Anytown')
    realtor_api_data = get_property_data_from_official_api(api_name='Realtor', location='Anytown')
    api_data_combined = zillow_api_data + realtor_api_data
    
    print("\n--- Combined API Data Loaded (Conceptual Official) ---")
    if api_data_combined:
        print(pd.DataFrame(api_data_combined).head())
    else:
        print("No API data to display.")

    if not crm_data.empty:
        unified_df = aggregate_real_estate_data(crm_data, scraped_data, api_data_combined)
        print("\n--- Final Unified Data Sample after Ingestion ---")
        print(unified_df.head())
        print(f"\nFinal unified data columns: {unified_df.columns.tolist()}")
    else:
        print("\nCRM data was not loaded. Cannot proceed with aggregation and preprocessing.")
        unified_df = pd.DataFrame()

    if not unified_df.empty:
        cleaned_df = perform_data_cleaning(unified_df.copy())
        print("\nCleaned DataFrame Head:")
        print(cleaned_df.head())
        cleaned_df.info()

        outlier_cols = ['monthly_rent', 'payment_delays_last_6_months', 'maintenance_requests_last_year', 'sqft', 'price']
        df_no_outliers = handle_outliers(cleaned_df.copy(), outlier_cols, strategy='cap')
        print("\nDataFrame after Outlier Handling (Capped) Head:")
        print(df_no_outliers.head())

        df_processed = drop_redundant_values_and_columns(df_no_outliers.copy())
        print("\nDataFrame after Redundancy Removal Head:")
        print(df_processed.head())
        print(f"Columns after redundancy removal: {df_processed.columns.tolist()}")

        statistical_analysis_summary(df_processed.copy())
        customer_behavior_summary(df_processed.copy())

        X_classification_final, y_classification_final, X_regression_final, y_regression_final, df_for_viz, scaler_classification, scaler_regression = \
            prepare_for_modeling(df_processed.copy(), target_column_classification='lease_renewal', target_column_regression='monthly_rent')
        
        print("\n--- Final Data for Modeling (Features X for Classification) Head ---")
        if X_classification_final is not None:
            print(X_classification_final.head())
        else:
            print("X_classification_final is None.")
        
        print("\n--- Final Data for Modeling (Target y for Classification) Head ---")
        if y_classification_final is not None:
            print(y_classification_final.head())
        else:
            print("y_classification_final is None, cannot display head.")

        print("\n--- Final Data for Modeling (Features X for Regression) Head ---")
        if X_regression_final is not None:
            print(X_regression_final.head())
        else:
            print("X_regression_final is None.")

        print("\n--- Final Data for Modeling (Target y for Regression) Head ---")
        if y_regression_final is not None:
            print(y_regression_final.head())
        else:
            print("y_regression_final is None, cannot display head.")

        print("\n--- Data for Visualization (df_for_viz) Head (with encoded features) ---")
        print(df_for_viz.head())

        trained_models = {}

        if y_classification_final is not None and not y_classification_final.empty:
            print("\n" + "="*50)
            print("Starting Hyperparameter Optimization for Random Forest Classifier")
            print("="*50)
            optimized_rf_classifier = perform_hyperparameter_optimization(
                X_classification_final, y_classification_final, 'classification', 'RandomForestClassifier'
            )
            if optimized_rf_classifier:
                trained_models['churn_rf_optimized'] = train_evaluate_classification_model(X_classification_final, y_classification_final, optimized_rf_classifier, "Optimized Random Forest Classifier")
            else:
                print("Skipping Optimized Random Forest Classifier: Optimization failed or data unavailable.")

            log_reg_model = LogisticRegression(random_state=42, max_iter=1000)
            trained_models['churn_log_reg'] = train_evaluate_classification_model(X_classification_final, y_classification_final, log_reg_model, "Logistic Regression")

            if tf is not None:
                trained_models['churn_nn'] = train_evaluate_neural_network_classification(X_classification_final, y_classification_final, "Neural Network (Tenant Churn)")
            else:
                print("\nSkipping Neural Network for Classification: TensorFlow not installed.")
        else:
            print("\nSkipping Tenant Churn Prediction models: Classification target data is not available.")

        if y_regression_final is not None and not y_regression_final.empty:
            print("\n" + "="*50)
            print("Starting Hyperparameter Optimization for Random Forest Regressor")
            print("="*50)
            optimized_rf_regressor = perform_hyperparameter_optimization(
                X_regression_final, y_regression_final, 'regression', 'RandomForestRegressor'
            )
            if optimized_rf_regressor:
                trained_models['rent_rf_optimized'] = train_evaluate_regression_model(X_regression_final, y_regression_final, optimized_rf_regressor, "Optimized Random Forest Regressor", target_name="Monthly Rent")
            else:
                print("Skipping Optimized Random Forest Regressor: Optimization failed or data unavailable.")
        else:
            print("\nSkipping Monthly Rent Prediction model: Regression target data is not available.")

        print("\n" + "="*50)
        print("--- Conceptual Model Deployment (using MLflow for registration) ---")
        print("="*50)

        if 'churn_rf_optimized' in trained_models and trained_models['churn_rf_optimized'] is not None:
            save_model(trained_models['churn_rf_optimized'], 'sklearn', 'Optimized Random Forest Classifier', scaler_classification)
        if 'churn_log_reg' in trained_models and trained_models['churn_log_reg'] is not None:
            save_model(trained_models['churn_log_reg'], 'sklearn', 'Logistic Regression', scaler_classification)
        if 'churn_nn' in trained_models and trained_models['churn_nn'] is not None:
            save_model(trained_models['churn_nn'], 'keras', 'Neural Network (Tenant Churn)', scaler_classification)
        if 'rent_rf_optimized' in trained_models and trained_models['rent_rf_optimized'] is not None:
            save_model(trained_models['rent_rf_optimized'], 'sklearn', 'Optimized Random Forest Regressor', scaler_regression)

        print("\n--- Simulating a Prediction with a Deployed Model from MLflow ---")
        if X_classification_final is not None and not X_classification_final.empty and 'churn_rf_optimized' in trained_models:
            sample_new_data = X_classification_final.iloc[[0]]
            print(f"\nSample new data for prediction:\n{sample_new_data}")
            
            mlflow_model_uri = "models:/Optimized Random Forest Classifier/latest"
            churn_prediction = load_model_and_predict(
                mlflow_model_uri,
                'deployed_models/Optimized_Random_Forest_Classifier_scaler.joblib',
                sample_new_data
            )
            if churn_prediction.size > 0:
                print(f"Predicted Tenant Churn for sample: {'Churn' if churn_prediction[0] == 0 else 'Renewal'}")
        else:
            print("Cannot simulate prediction: No classification model deployed or data unavailable.")

        print("\n" + "="*50)
        print("--- Conceptual Model Monitoring (leveraging MLflow for baselines) ---")
        print("="*50)
        if 'churn_rf_optimized' in trained_models and X_classification_final is not None and y_classification_final is not None and scaler_classification is not None:
            simulate_monitoring(
                trained_models['churn_rf_optimized'],
                "Optimized Random Forest Classifier (Churn)",
                X_classification_final,
                y_classification_final,
                scaler_classification,
                prediction_type='classification'
            )
        else:
            print("Cannot simulate monitoring: Optimized Random Forest Classifier for churn not available or scaler missing.")

        if 'rent_rf_optimized' in trained_models and X_regression_final is not None and y_regression_final is not None and scaler_regression is not None:
            simulate_monitoring(
                trained_models['rent_rf_optimized'],
                "Optimized Random Forest Regressor (Rent)",
                X_regression_final,
                y_regression_final,
                scaler_regression,
                prediction_type='regression'
            )
        else:
            print("Cannot simulate monitoring: Optimized Random Forest Regressor for rent not available or scaler missing.")

        print("\n" + "="*50)
        print("--- Conceptual Iteration and Continuous Improvement ---")
        print("="*50)
        conceptual_iteration_loop()

    else:
        print("\nUnified DataFrame is empty. Skipping preprocessing, analysis, and ML modeling, deployment, and monitoring.")