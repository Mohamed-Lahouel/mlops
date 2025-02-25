# Variables
TRAIN_FILE = churn-bigml-80.csv
TEST_FILE = churn-bigml-20.csv
MODEL_FILE = model.pkl
MAIN_SCRIPT = main.py

# Targets
.PHONY: all prepare train evaluate run clean start-api mlflow-start mlflow-ui

# Default target
all: run

# Prepare the data (loading and preprocessing)
prepare:
	python $(MAIN_SCRIPT) --prepare

# Train the model and save it, logging to MLflow
train:
	python $(MAIN_SCRIPT) --train

# Evaluate the model on test data, logging results to MLflow
evaluate:
	python $(MAIN_SCRIPT) --evaluate

# Run the full pipeline (train, evaluate, and save the model), logging all to MLflow
run:
	python $(MAIN_SCRIPT) --run

# Clean generated files (e.g., model file)
clean:
	rm -f $(MODEL_FILE)

# Start FastAPI app for prediction
start-api:
	uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Start MLflow server (tracking API and UI)
mlflow-start:
	mlflow server --host 0.0.0.0 --port 5000

# Open MLflow UI (optional, assuming MLflow server is already running)
mlflow-ui:
	open http://localhost:5000
