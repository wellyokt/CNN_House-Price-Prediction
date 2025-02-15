import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from PIL import Image
import cv2
import time
from samgeo import tms_to_geotiff
from src.utils.logger import default_logger as logger
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Dense, Concatenate, GlobalAveragePooling2D

from io import BytesIO
import base64
from shapely.geometry import mapping
import os
import sys
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)



class Preprocessor:
    def __init__(self, data, scaler):
        self.data = data
        self.scaler = scaler
        self.gdf_data =None
        self.geom_buffer =None
        self.image =None
        self.X_transaction_scaled =None
    
    def gdf(self, data):
        """Preprocessing data for predition"""
        # try:
        data['geometry'] =data.apply(lambda x:Point(x['longitude'],x['latitude']),axis=1)
        self.gdf_data = gpd.GeoDataFrame(data, geometry='geometry', crs=4326)
        # except Exception as e:
        #     logger.error(f"failed to preprocess data")
        #     raise

    def buffer_area(self):
        """Buffering area of the geometry"""
        try:
            geom_buffer= self.gdf_data.to_crs(3393).buffer(1000, cap_style='square')
            self.geom_buffer = gpd.GeoDataFrame(geom_buffer,geometry =geom_buffer).to_crs(4326)
        except Exception as e:
            logger.error(f"failed to buffer area")
            raise
    
    def satelit_image(self):
        """Getting satellite image data"""
        try:
            # get satellite image data
            bbox =list(self.geom_buffer['geometry'][0] .bounds)
            random = int(time.time() * 1000)
     
            #get image
            output_dir =os.path.join(project_root, 'artifacts', 'data_prediction')
            tiff_image_path = os.path.join(output_dir,f"image_{random}.tiff")
            tms_to_geotiff(output=tiff_image_path, bbox=bbox, zoom=17, source="Satellite", overwrite=True)
            self.png_image_path = os.path.join(output_dir,f"image_{random}.png")

            # Open the TIFF file using Pillow
            with Image.open(tiff_image_path) as img:
                # Save the image as PNG
                img.save(self.png_image_path, format="PNG")

        except Exception as e:
            logger.error(f"failed to get satellite image")
            raise
    # Function to load and preprocess images
    def load_images(self, img_size=(224, 224)):
        """Loading and preprocessing images"""
        try:
            # Ensure the image file exists
  
            img = cv2.imread(self.png_image_path)
            img = cv2.resize(img, img_size)
            img = img / 255.0  # Normalize
            self.img = np.array(img)

    
        except Exception as e:
            logger.error(f"failed to load image")
            raise
     
    def fit_transform(self):
        """Scaling data using minmaxscaler"""
        try:
            
            selected_features = ['bedroom', 'bathroom', 'building_area', 'land_area']
            X_transaction = self.gdf_data[selected_features].fillna(0).values
            self.X_transaction_scaled = self.scaler.transform(X_transaction)
 
        except Exception as e:
            logger.error(f"failed to scale data")
            raise
        


    def extract_feature(self):
        """Extracting features from transaction data"""
        try:
            if self.gdf_data is None:
                self.gdf(self.data)
            if self.geom_buffer is None:
                self.buffer_area()
            if self.image is None:
                self.satelit_image()
                self.load_images()
            if self.X_transaction_scaled is None:
                self.fit_transform()
            
            X_img= np.expand_dims(self.img, axis=0)  # Shape becomes (1, 224, 224, 3)
            # X_tx = np.expand_dims(self.X_transaction_scaled, axis=0) 

            return X_img,self.X_transaction_scaled,self.geom_buffer
        except Exception as e:
            logger.info(f"Failed Extract Feature: {e}")
            raise
    # Function to Convert Image to Base64 for JSON Response
    def encode_image(self,image_array):
        """Convert NumPy array image to Base64 string."""
        image = Image.fromarray((image_array * 255).astype(np.uint8))  # Convert to uint8 image
        buffered = BytesIO()
        image.save(buffered, format="PNG")  # Save as PNG
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

            
    def convert_geometry(self,geometry):
        """Convert Shapely Geometry to JSON-serializable format."""
        if geometry is None:
            return None
        return mapping(geometry)  # Convert to GeoJSON format



