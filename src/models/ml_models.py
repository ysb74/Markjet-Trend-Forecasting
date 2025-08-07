import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score


# Import TensorFlow and Keras for Neural Networks
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    print("TensorFlow not installed. Neural Network models will be skipped.")
    tf = None

# Import MLflow for MLOps capabilities
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.tensorflow
except ImportError:
    print("MLflow not installed. MLOps features will be skipped.")
    mlflow = None

# Import plotting and evaluation utilities
from src.utils.evaluation import plot_confusion_matrix, plot_actual_vs_predicted, plot_training_history


def train_evaluate_classification_model(X, y, model, model_name):
    """
    Trains and evaluates a classification model. Logs results to MLflow.
    """
    print(f"\n--- Training and Evaluating {model_name} for Tenant Churn Prediction ---")
    if X is None or y is None or X.empty:
        print(f"Skipping {model_name}: X or y is None or empty.")
        return None
    
    if mlflow is not None:
        with mlflow.start_run(run_name=f"{model_name}_Training_Classification"):
            mlflow.set_tag("model_type", "classification")
            mlflow.set_tag("task", "tenant_churn_prediction")
            
            if hasattr(model, 'get_params'):
                mlflow.log_params(model.get_params())

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=['Churn', 'Renewal'], output_dict=True)
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision_churn", report['Churn']['precision'])
            mlflow.log_metric("recall_churn", report['Churn']['recall'])
            mlflow.log_metric("f1-score_churn", report['Churn']['f1-score'])
            mlflow.log_metric("precision_renewal", report['Renewal']['precision'])
            mlflow.log_metric("recall_renewal", report['Renewal']['recall'])
            mlflow.log_metric("f1-score_renewal", report['Renewal']['f1-score'])
            
            print(f"Accuracy ({model_name}): {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['Churn', 'Renewal']))

            plot_confusion_matrix(y_test, y_pred, model_name, labels=['Churn', 'Renewal'])

            mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)
            print(f"Model '{model_name}' logged to MLflow.")
            return model
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy ({model_name}): {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Churn', 'Renewal']))
        plot_confusion_matrix(y_test, y_pred, model_name, labels=['Churn', 'Renewal'])
        return model


def train_evaluate_regression_model(X, y, model, model_name, target_name="Target"):
    """
    Trains and evaluates a regression model. Logs results to MLflow.
    """
    print(f"\n--- Training and Evaluating {model_name} for {target_name} Prediction ---")
    if X is None or y is None or X.empty:
        print(f"Skipping {model_name}: X or y is None or empty.")
        return None

    if mlflow is not None:
        with mlflow.start_run(run_name=f"{model_name}_Training_Regression"):
            mlflow.set_tag("model_type", "regression")
            mlflow.set_tag("task", f"{target_name}_prediction")

            if hasattr(model, 'get_params'):
                mlflow.log_params(model.get_params())

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2_score", r2)

            print(f"Mean Squared Error ({model_name}): {mse:.4f}")
            print(f"R-squared ({model_name}): {r2:.4f}")

            plot_actual_vs_predicted(y_test, y_pred, model_name, target_name)

            mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)
            print(f"Model '{model_name}' logged to MLflow.")
            return model
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Mean Squared Error ({model_name}): {mse:.4f}")
        print(f"R-squared ({model_name}): {r2:.4f}")
        plot_actual_vs_predicted(y_test, y_pred, model_name, target_name)
        return model


def train_evaluate_neural_network_classification(X, y, model_name="Neural Network"):
    """
    Trains and evaluates a simple Neural Network for classification. Logs results to MLflow.
    """
    if tf is None:
        print(f"Skipping {model_name}: TensorFlow not installed.")
        return None
    if X is None or y is None or X.empty:
        print(f"Skipping {model_name}: X or y is None or empty.")
        return None

    print(f"\n--- Training and Evaluating {model_name} for Tenant Churn Prediction ---")

    if mlflow is not None:
        with mlflow.start_run(run_name=f"{model_name}_Training_Classification"):
            mlflow.set_tag("model_type", "classification")
            mlflow.set_tag("task", "tenant_churn_prediction")
            mlflow.log_param("epochs", 100)
            mlflow.log_param("batch_size", 32)
            mlflow.log_param("optimizer", "adam")
            mlflow.log_param("loss_function", "binary_crossentropy")
            mlflow.log_param("layer_1_neurons", 64)
            mlflow.log_param("layer_2_neurons", 32)
            mlflow.log_param("layer_3_neurons", 16)
            mlflow.log_param("layer_4_neurons", 8)
            mlflow.log_param("dropout_rate", 0.3)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

            model = keras.Sequential([
                layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(16, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(8, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(1, activation='sigmoid')
            ])

            model.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

            history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            print(f"Test Accuracy ({model_name}): {accuracy:.4f}")

            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_loss", loss)
            mlflow.log_metrics({"train_accuracy_final": history.history['accuracy'][-1],
                                 "val_accuracy_final": history.history['val_accuracy'][-1],
                                 "train_loss_final": history.history['loss'][-1],
                                 "val_loss_final": history.history['val_loss'][-1]})

            y_pred_proba = model.predict(X_test, verbose=0).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)

            print("\nClassification Report:")
            report = classification_report(y_test, y_pred, target_names=['Churn', 'Renewal'], output_dict=True)
            print(classification_report(y_test, y_pred, target_names=['Churn', 'Renewal']))
            mlflow.log_metric("precision_churn", report['Churn']['precision'])
            mlflow.log_metric("recall_churn", report['Churn']['recall'])
            mlflow.log_metric("f1-score_churn", report['Churn']['f1-score'])
            mlflow.log_metric("precision_renewal", report['Renewal']['precision'])
            mlflow.log_metric("recall_renewal", report['Renewal']['recall'])
            mlflow.log_metric("f1-score_renewal", report['Renewal']['f1-score'])

            plot_confusion_matrix(y_test, y_pred, model_name)
            plot_training_history(history, model_name)

            mlflow.tensorflow.log_model(model, "model", registered_model_name=model_name)
            print(f"Model '{model_name}' logged to MLflow.")
            return model
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(8, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy ({model_name}): {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, (model.predict(X_test) > 0.5).astype(int), target_names=['Churn', 'Renewal']))
        plot_confusion_matrix(y_test, (model.predict(X_test) > 0.5).astype(int), model_name)
        plot_training_history(history, model_name)
        return model


def perform_hyperparameter_optimization(X_train, y_train, model_type, model_name):
    """
    Performs hyperparameter optimization using GridSearchCV for a given model type.
    Logs the best parameters and score to MLflow.
    """
    print(f"\n--- Performing Hyperparameter Optimization for {model_name} ---")
    if X_train is None or y_train is None or X_train.empty:
        print(f"Skipping optimization for {model_name}: Training data is None or empty.")
        return None

    if model_type == 'classification':
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        scoring = 'accuracy'
    elif model_type == 'regression':
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        scoring = 'neg_mean_squared_error'
    else:
        print(f"Invalid model_type: {model_type}. Must be 'classification' or 'regression'.")
        return None

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                               scoring=scoring, cv=3, n_jobs=-1, verbose=1)
    
    grid_search.fit(X_train, y_train)

    print(f"\nBest parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best cross-validation score for {model_name}: {grid_search.best_score_:.4f}")

    if mlflow is not None:
        with mlflow.start_run(run_name=f"{model_name}_HPO"):
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric(f"best_cv_score_{scoring}", grid_search.best_score_)
            mlflow.set_tag("hpo_model", model_name)
            mlflow.set_tag("hpo_scoring", scoring)

    return grid_search.best_estimator_