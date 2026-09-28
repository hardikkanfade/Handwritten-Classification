import os
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import tensorflow as tf

from backend.preprocess import preprocess_image

app = FastAPI(title="DigitVision API")

# Enable CORS so frontend can call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

@app.on_event("startup")
def load_model():
    """Load the model once at startup."""
    global model
    model_path = os.path.join(os.path.dirname(__file__), "..", "model", "mnist_cnn.keras")
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
    else:
        print("Warning: Model not found. Run train_model.py to train it.")

class PredictRequest(BaseModel):
    image: str

@app.post("/predict")
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train the model first.")
    
    try:
        # Preprocess the base64 canvas image into a 28x28 normalized tensor
        processed_img = preprocess_image(request.image)
        
        # Predict
        preds = model.predict(processed_img, verbose=0)[0]

        # Extract results
        digit = int(np.argmax(preds))
        confidence = float(np.max(preds))
        probabilities = [float(p) for p in preds]

        return {
            "digit": digit,
            "confidence": confidence,
            "probabilities": probabilities
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

# Serve the frontend as static files on the root URL
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
