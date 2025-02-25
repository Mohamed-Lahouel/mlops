import joblib
import argparse
import pandas as pd
import os
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, roc_curve, auc, confusion_matrix, precision_recall_curve
from model_pipeline import load_data, preprocess_data, train_classifier, evaluate_classifier, save_model

# Configuration
TRAIN_FILE = "churn-bigml-80.csv"
TEST_FILE = "churn-bigml-20.csv"
MODEL_FILE = "model.pkl"

# Set MLflow Tracking URI and experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mohamedexperiences")

def prepare():
    """Loads and preprocesses the data."""
    train_data, test_data = load_data(TRAIN_FILE, TEST_FILE)
    X_train, y_train, X_test, y_test = preprocess_data(train_data, test_data)
    print("Data preparation completed!")
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

def train():
    """Trains and saves the model."""
    train_data, test_data = load_data(TRAIN_FILE, TEST_FILE)
    X_train, y_train, X_test, y_test = preprocess_data(train_data, test_data)

    with mlflow.start_run():
        model = train_classifier(X_train, y_train)
        save_model(model, MODEL_FILE)

        # Log model parameters
        mlflow.log_param("iterations", 100)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("depth", 3)

        # Evaluate model
        accuracy, auc_score, report = evaluate_classifier(model, X_test, y_test)

        # Additional Metrics
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        logloss = log_loss(y_test, y_pred_proba)

        print("Model trained successfully!")
        print(f"Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, Log Loss: {logloss:.4f}")

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("auc", auc_score)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("log_loss", logloss)

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        # ROC Curve with shaded area
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color="blue", label=f"ROC curve (area = {roc_auc:.2f})")
        plt.fill_between(fpr, tpr, color='blue', alpha=0.1)  # Shading under the curve
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.savefig("roc_curve.png")
        mlflow.log_artifact("roc_curve.png")

        # Precision-Recall Curve with style change
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.figure()
        plt.plot(recall_vals, precision_vals, color="green", label="Precision-Recall Curve", linestyle="--", linewidth=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.savefig("precision_recall_curve.png")
        mlflow.log_artifact("precision_recall_curve.png")

        # Feature Importance Plot (if applicable)
        try:
            feature_importance = model.feature_importances_
            features = X_train.columns
            plt.figure(figsize=(10, 6))
            plt.barh(features, feature_importance, color="orange")
            plt.xlabel("Feature Importance")
            plt.title("Feature Importance")
            plt.savefig("feature_importance.png")
            mlflow.log_artifact("feature_importance.png")
        except AttributeError:
            print("Feature importance not available for the model type.")

        # Log and Register Model
        mlflow.sklearn.log_model(model, "model", registered_model_name="Customer_Churn_Model")

        print("Model registered successfully!")

def evaluate():
    """Loads and evaluates the model."""
    if not os.path.exists(MODEL_FILE):
        print("Model file not found! Train the model first.")
        return

    train_data, test_data = load_data(TRAIN_FILE, TEST_FILE)
    _, _, X_test, y_test = preprocess_data(train_data, test_data)

    model = joblib.load(MODEL_FILE)
    accuracy, auc_score, report = evaluate_classifier(model, X_test, y_test)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)

    print("Model evaluation completed!")
    print(f"Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, Log Loss: {logloss:.4f}")

def clean():
    """Removes the model file."""
    if os.path.exists(MODEL_FILE):
        os.remove(MODEL_FILE)
        print("Model file cleaned up!")
    else:
        print("No model file to clean.")

def run():
    """Run the full pipeline (prepare, train, evaluate, and clean)."""
    prepare()
    train()
    evaluate()
    clean()

# Command-line interface (CLI) argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Pipeline for Churn Prediction")
    parser.add_argument("--prepare", action="store_true", help="Prepare the data")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--clean", action="store_true", help="Clean the model file")
    parser.add_argument("--run", action="store_true", help="Run the full pipeline")

    args = parser.parse_args()

    if args.prepare:
        prepare()
    elif args.train:
        train()
    elif args.evaluate:
        evaluate()
    elif args.clean:
        clean()
    elif args.run:
        run()
    else:
        print("No action specified. Use --help for usage instructions.")

