from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import mlflow
import mlflow.sklearn

# Initialize the FastAPI app
app = FastAPI()

# Load the model at the start of the app
MODEL_PATH = "model.pkl"  # Corrected model file name
model = None
try:
    model = joblib.load(MODEL_PATH)  # Load the model
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {str(e)}")

# Define a schema for the request body with Pydantic
class PredictionInput(BaseModel):
    State: int
    Account_length: int
    International_plan: int
    Voice_mail_plan: int
    Total_day_minutes: float
    Total_day_calls: int
    Total_eve_minutes: float
    Total_eve_calls: int
    Total_night_minutes: float
    Total_night_calls: int
    Total_intl_minutes: float
    Total_intl_calls: int
    Customer_service_calls: int

# Define the prediction route
@app.post("/predict")
def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly")

    try:
        # Convert the input data into a numpy array (reshape to make it a 2D array)
        input_array = np.array([
            input_data.State,
            input_data.Account_length,
            input_data.International_plan,
            input_data.Voice_mail_plan,
            input_data.Total_day_minutes,
            input_data.Total_day_calls,
            input_data.Total_eve_minutes,
            input_data.Total_eve_calls,
            input_data.Total_night_minutes,
            input_data.Total_night_calls,
            input_data.Total_intl_minutes,
            input_data.Total_intl_calls,
            input_data.Customer_service_calls
        ]).reshape(1, -1)  # Reshape into 1 row with multiple columns for prediction

        # Perform the prediction
        prediction = model.predict(input_array)
        result = bool(prediction[0])  # Convert to a native Python boolean if needed

        # Log the input and result with MLflow
        with mlflow.start_run():
            mlflow.log_params(input_data.dict())  # Log the input features
            mlflow.log_metric("prediction", result)  # Log the prediction result

        return {"prediction": result}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

# Optional: Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the prediction API"}

