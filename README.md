## **Project Overview: House Price Prediction with Deep Learning**
This project predicts property prices using a multimodal deep learning model that combines satellite imagery and structured transactional data.



**Satellite Image Features** : Extracted using VGG16 to capture geographic attributes.
**Property Attributes**: Bedrooms, bathrooms, building area, and land area.
**Location Coordinates**: Latitude and longitude to capture spatial context.

How does the model work?
The model uses a multimodal approach:

Image Path (VGG16): Processes satellite images.
Structured Data Path (FCNN): Processes property features.
Fusion Layer: Combines insights from both paths.
Output Layer: Predicts the property price.

## **Key Features of the Application**:
⚡ Real-time Predictions: Input property details for instant estimates.

## Project Structure
```plaintext
boston_house_price/
│
├── .streamlit/                      # Streamlit configuration
│   └── config.toml                  # Streamlit settings
│
├── artifacts/                       # Model artifacts
│   ├── data                         # Dataset
│   ├── data_prediction               # data saved when api call
│   ├── model                         # model an scaler
│
│
├── logs/                           # Application logs
│   └── app.log
│
├── apps_streamlit/                 # Streamlit pages
│   ├── Home.py                     #Streamlit main page
│   ├── 2_my_modul.py               # module for streamlit apps
│
├── src/                            # Source code
│   ├── api                          #main apps
│   ├── data                        #Preprocessing data
│   ├── model                       # Model training
│   └── utils                       #logging setup
│
├── app.py                          # FastAPI application
├── requirements.txt                # Dependencies
|── Vercel.json                     #json for deploy api using vercel
└── README.md                       # Documentation
```



### Local Development Setup

1. Clone the repository:
```bash
git https://github.com/wellyokt/CNN_House-Price-Prediction.git
cd CNN_House-Price-Prediction
```

2. Create and activate virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# For Windows
.\venv\Scripts\activate
# For Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Run the applications:
```bash
# Terminal 1 - Run FastAPI
uvicorn app:app --reload --port 8000

# Terminal 2 - Run Streamlit
streamlit run Home.py
``'