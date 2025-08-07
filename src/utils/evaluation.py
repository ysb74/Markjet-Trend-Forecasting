import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
import os
import mlflow # For logging plots as artifacts

def plot_confusion_matrix(y_true, y_pred, model_name, labels=['Churn', 'Renewal']):
    """
    Plots and saves a confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Predicted {labels[0]}', f'Predicted {labels[1]}'],
                yticklabels=[f'Actual {labels[0]}', f'Actual {labels[1]}'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    plot_path = f"confusion_matrix_{model_name.replace(' ', '_')}.png"
    plt.savefig(plot_path)
    if mlflow is not None:
        mlflow.log_artifact(plot_path)
    plt.show()
    os.remove(plot_path) # Clean up the plot file

def plot_actual_vs_predicted(y_true, y_pred, model_name, target_name="Target"):
    """
    Plots and saves actual vs. predicted values for regression.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.title(f'Actual vs. Predicted {target_name} ({model_name})')
    plt.xlabel(f'Actual {target_name}')
    plt.ylabel(f'Predicted {target_name}')
    plt.grid(True)
    plt.legend()

    plot_path = f"actual_vs_predicted_{model_name.replace(' ', '_')}.png"
    plt.savefig(plot_path)
    if mlflow is not None:
        mlflow.log_artifact(plot_path)
    plt.show()
    os.remove(plot_path) # Clean up the plot file

def plot_training_history(history, model_name):
    """
    Plots and saves the training history (accuracy and loss) for Keras models.
    """
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()

    plot_path = f"training_history_{model_name.replace(' ', '_')}.png"
    plt.savefig(plot_path)
    if mlflow is not None:
        mlflow.log_artifact(plot_path)
    plt.show()
    os.remove(plot_path) # Clean up the plot file