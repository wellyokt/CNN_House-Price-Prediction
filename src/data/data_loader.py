import pandas as pd
import geopandas as gpd
from typing import Optional, Tuple
import pickle
from shapely.geometry import Point
from src.utils.logger import default_logger as logger

class DataLoader:
    def __init__(self, model_path: Optional[str] = None,scaler_path: Optional[str] = None):
        """
        Initialize data loader
        
        Args:
            data_path: Optional path to data file
        """
        self.model_path = model_path 
        self.scaler_path = scaler_path 
        logger.info(f"Initialized model with path: {self.model_path}")
        logger.info(f"Initialized scaler with path: {self.scaler_path}")



    def load_model(self):
        """
        Load the model from the specified path
        """
        try:
            with open(self.model_path, 'rb') as file:
                model =pickle.load(file)  
            logger.info(f"model load succesfully")
            return model
        except Exception as e:
            logger.error(f"failed to load model: {e}")
            raise
    
    
        
    def load_scaler(self):
        """Load the scaler"""
        try:
            with open(self.scaler_path, 'rb') as file:
                scaler =pickle.load(file)  
            logger.info(f"Load scaller succes")
            return scaler
        except Exception as e:
            logger.error(f"failed to load scaler: {e}")
            raise


        
    def load_data_prediction(self,json):
        """Load the data prediction
        args:
        json Json object containing the prediction data."""
        try:
            df =pd.DataFrame(json.dict(), index=[0])
            logger.info(f"Load data succes")

            return df
        except Exception as e:
            logger.error(f"failed to load data: {e}")
            raise