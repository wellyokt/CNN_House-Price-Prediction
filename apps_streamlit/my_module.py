import streamlit as st
import pandas as pd


import json
import re
import requests
import os
import base64
import streamlit as st
from io import BytesIO
from PIL import Image


def decode_base64_image(base64_string):
    """Convert Base64 string back to an image."""
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_bytes))  # Convert to PIL Image
    return image

@st.cache_data(ttl=3600)
def hide_button():
    st.markdown("""
    <style>
        #root > div:nth-child(1) > div > div > div > div > section > div {
            padding-top: 2rem;
            padding-bottom: 0rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display:none;}
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_data():
    """Load and cache the dataset"""
    return pd.read_csv(Config.DATA_PATH)


@st.cache_data(ttl=3600)
def load_model_artifacts():
    """Load and cache model metrics and feature importance"""
    try:
        with open(Config.METRICS_PATH, 'r') as f:
            metrics = json.load(f)
        
        with open(Config.FEATURE_IMPORTANCE_PATH, 'r') as f:
            feature_importance = json.load(f)
                
        return metrics, feature_importance
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None

