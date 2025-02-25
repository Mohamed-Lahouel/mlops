import pandas as pd
import joblib
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Load the dataset
def load_data(train_file, test_file):
    """Loads train and test datasets."""
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    return train_df, test_df

# Preprocess data
def preprocess_data(train_df, test_df):
    """Splits data into features (X) and target (y)."""
    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"]
    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]
    return X_train, y_train, X_test, y_test

# Train the model
def train_classifier(X_train, y_train):
    """Trains a CatBoost classifier."""
    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=3,
        random_state=42,
        verbose=False
    )
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_classifier(model, X_test, y_test):
    """Evaluates the model on test data."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, auc, report

# Save the model
def save_model(model, filename="model.pkl"):
    """Saves the trained model."""
    joblib.dump(model, filename)

