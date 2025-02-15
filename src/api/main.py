from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import uvicorn
import pandas as pd
import sys
import os
import numpy as np




# Adjust the Python path to include the `src` directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.logger import default_logger as logger
from src.data.data_preprocessor import Preprocessor
from src.data.data_loader import DataLoader

app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting house prices based on several features.",
    version="1.0.0"
)

# Pydantic model for data validation
class HouseData(BaseModel):
    bedroom: int
    bathroom: int
    building_area: int
    land_area: int
    latitude: float
    longitude: float

class PredictionResponse(BaseModel):
    price_prediction: int
    latitude: float
    longitude: float
    image: str

# DataLoader instance
model_path = r"D:\ML5\final_projectt\artifacts\model\model.pkl"
scaler_path = r"D:\ML5\final_projectt\artifacts\model\scaler.pkl"
dl = DataLoader(model_path, scaler_path)



# Load model and scaler on startup
@app.on_event("startup")
async def load_model_scaler():
    global model, scaler, data
    model = dl.load_model()
    scaler = dl.load_scaler()

    #printjhdd12
@app.post("/predict", response_model=int)
async def predict(house_data: HouseData):
    logger.info(f"Received data for prediction: {house_data.json()}")

    # Prepare data for prediction
    data = dl.load_data_prediction(house_data)
    data_processor = Preprocessor(data,scaler)
    x_img, x_tx, geom_buffer= data_processor.extract_feature()
    

    # Predict
    try:
        prediction = model.predict([x_img,x_tx]) # Adjust indexing based on your model's output
        prediction =int(prediction[0][0])
        return prediction 
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Model prediction failed")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)