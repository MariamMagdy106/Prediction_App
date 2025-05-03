import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import gc
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import List, Union
from pathlib import Path
import uvicorn



WINDOW_SIZE_MS = int(1*60*1000)
MODEL_DIR = Path("models") # Directory containing model files
SCALER_DIR = Path("scalers") # Directory containing model files

app = FastAPI()

# Define the input data model
class PredictionRequest(BaseModel):
    """Request model for prediction endpoint.
    
    Attributes:
        model_name (str): Name of the model file (without extension)
        data (List[List[Union[str, float]]]): Input data as list of lists containing 
            timestamp (str) and feature values (float)
    """
    model_name: str
    data: List[List[Union [str ,float]]]


def last_leq_index(arr: np.ndarray, start_index: int, value: int) -> int:
    """Find the last index in array where value is <= target value."""
    for i in range(start_index+1,  len(arr)):
        if arr[i] > value:
            return i-1
    return -1

def split_sequence_X(sequence, timestamps, window_size_ms):
    """
    Splits time-series data into fixed-duration windows.
    
    Args:
        sequence: List/array of feature values (shape: [n_samples, n_features])
        timestamps: Array of corresponding timestamps in milliseconds (shape: [n_samples])
        window_size_ms: Window duration in milliseconds
        
    Returns:
        List of windows (each window: array of shape [window_length, n_features])
        
    Raises:
        ValueError: If input data cannot create any valid windows
    """
    X = []
    n = len(sequence)
    
    for i in range(n):
        end_ix = last_leq_index(timestamps, i, timestamps[i] + window_size_ms)
        if end_ix == -1:
            break  # No more valid windows
        X.append(sequence[i:end_ix])
    
    if not X:  # Check for empty output
        raise ValueError(
            f"No valid windows created. Required window size: {window_size_ms/1000}s, "
            f"but input data only covers {(timestamps[-1]-timestamps[0])/1000}s"
        )
    return X
def preprocess_data(data: List[List[Union[str, float]]]) -> np.ndarray:
    """Preprocess input data for model prediction."""
    # Convert the data to a NumPy array
    data_array = np.array(data)
    timestamps = (pd.to_datetime(data_array[:, 0]).astype("int64") / 10**6).astype("int64")
    sequence = data_array[:, 1:].astype("float32")

    # Generate sequences
    X = split_sequence_X(sequence,timestamps, WINDOW_SIZE_MS)
    return pad_sequences(X, padding='post', dtype='float32')
    

def f1_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Custom F1 score loss function for model training."""
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

def load_model_and_scaler(model_path: Path, scaler_path: Path) -> tuple:
    """Load trained model and scaler from disk."""
    try:
        model = tf.keras.models.load_model(model_path, custom_objects={'f1_loss': f1_loss})
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model or scaler: {str(e)}")
    
models_and_scalers = {}    
def load_model_and_scaler_cached(model_name: str) -> tuple:
    
    if model_name in models_and_scalers:
        return models_and_scalers[model_name]
    
    model_path = MODEL_DIR / f"{model_name}.h5"
    scaler_path = SCALER_DIR / f"{model_name}.pkl"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise HTTPException(status_code=404, detail="Model or scaler not found")
    
    models_and_scalers[model_name] = load_model_and_scaler(model_path, scaler_path)
    return models_and_scalers[model_name]

# Define the prediction function
def predict_failure(model: tf.keras.Model, scaler: object, data: List[List[Union[str, float]]]) -> bool:
    """Make failure prediction using the loaded model."""
    data = preprocess_data(data)
    
    n_samples_test = data.shape[0]
    n_steps_test = data.shape[1]
    n_features_test = data.shape[2]

    data_scaled = scaler.transform(data.reshape((n_samples_test * n_steps_test, n_features_test)))
    data_scaled = data_scaled.reshape(n_samples_test, n_steps_test, n_features_test)

    prediction =  any(np.rint(model.predict(data_scaled)))
    return prediction

@app.post("/predict/")
async def predict(prediction_request: PredictionRequest) -> dict:
    """Prediction endpoint for failure detection."""


    model, scaler = load_model_and_scaler_cached(prediction_request.model_name)

    # Make prediction
    try:
        prediction = predict_failure(model, scaler, prediction_request.data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

    return {"prediction": prediction}

# Run the FastAPI app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
