    
import  streamlit_antd_components as sac
import streamlit as st
import pandas as pd
import numpy as np
from my_module import *
import folium
import sys
from shapely.geometry import Point
import geopandas as gpd
from streamlit_folium import folium_static
from samgeo import SamGeo, show_image, download_file, overlay_images, tms_to_geotiff


import os
from pathlib import Path



# Set page

st.set_page_config(page_title='House Prediction', page_icon=f"./data/Image_ic.ico", layout='wide')

# Hide button at the top right of the page
hide_button()



# Add the project root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

# Now, import the module
from src.data.data_preprocessor import Preprocessor
from src.data.data_preprocessor import Preprocessor
from src.data.data_loader import DataLoader
with st.sidebar:

# Menu in Sidebar
    item = sac.menu([
        sac.MenuItem('Profile', icon="bi bi-person-lines-fill"),
        sac.MenuItem('House Price Prediction', disabled=True),
        sac.MenuItem('Overview', icon='house-fill'),
        sac.MenuItem('Predictions', icon="bi bi-search")],
         format_func='title', size='md',open_all=False,color='blue',indent=15, open_index=0, )

if item =='Overview':
 
    # Project Overview
    st.markdown("""
    ### **Project Overview: Ads Click Prediction with Machine Learning**

    Welcome to the **Ads Click Prediction Application**! In today’s digital age, understanding user behavior and predicting the likelihood of an advertisement being clicked is crucial for optimizing online advertising campaigns. This application uses advanced machine learning techniques to accurately predict the likelihood of an advertisement being clicked based on various user characteristics and behavior patterns.

    ### **What does this application do?**

    This application predicts whether an advertisement will be clicked by a user based on a variety of user-specific factors. By analyzing features such as:

    - **Time Spent on Site**: The duration a user spends on the website, which can indicate their level of engagement.
    - **Age**: The user's age, which may influence the type of content or ads they find appealing.
    - **Area Income**: The income level of the user's geographical region, helping to tailor ads based on socioeconomic factors.
    - **Daily Internet Usage**: The average number of hours a user spends online, which can help identify more frequent internet users who may be more likely to interact with ads.
    - **Gender**: Whether the user is male or female, which may influence the ads they are most likely to interact with.
    - **Device Information**: Data regarding the device used to access the site, which could affect how ads are presented.

    ### **How does the model work?**

    Using these features, we train a machine learning model to classify user interactions with displayed ads. The model learns from historical data, including past user clicks and interactions with ads, to identify patterns and predict whether a user will click on an ad.

    Our model applies **Logistic Regression**, a simple and interpretable classification algorithm that predicts the probability of an ad click. Logistic regression is well-suited for binary classification tasks, such as predicting whether a user will or will not click an ad, and provides probabilities that help quantify the likelihood of interaction.


    ### **Key Features of the Application:**

    - **Real-time predictions**: Input your user data, and get an immediate prediction on whether an ad is likely to be clicked or not.
    - **Visualization**: Interactive charts and graphs to help users understand the feature importance and prediction outcomes.
    - **Model Performance**: A detailed performance evaluation of the machine learning model with metrics like accuracy, precision, recall, and F1-score, ensuring that the predictions are trustworthy.
    - **Feature Importance**: Visual representation of the most important features influencing the model's predictions (e.g., age, daily internet usage, etc.), allowing users to understand the factors driving ad clicks.

    ### **What will you find in this application?**

    - **Data Exploration**: An exploration of the user behavior dataset, visualizations of feature distributions, and correlation analysis.
    - **Model Training**: Training and testing machine learning models to predict ad clicks with metrics for evaluation.
    - **Prediction Results**: Instant predictions based on user input, helping businesses make data-driven decisions in their ad strategies.
    - **Performance Metrics**: Key model performance metrics such as accuracy, precision, recall, and F1 score, ensuring transparency in model reliability.

    """)


if item =='Profile':
    BASE_DIR = Path(__file__).parent.parent

    CV_DIR = BASE_DIR/'cv'
    #Profile
    FOTO_PATH = CV_DIR / "IMG_0236.png"
    project1_image = CV_DIR / "google-adsense-moves-to-pay-per-impression.png"


  
    st.markdown(
    """
    <div style="display: flex; justify-content: center;">
        <h1> Welcome to my portfolio </h1>
    </div>
    """,
    unsafe_allow_html=True
    )
    


    # CSS pour l'image ronde et le conteneur
    st.markdown(
        """
        <style>
        .round-img {
            border-radius: 50%;
            width: 400px; /* Ajustez la taille selon vos besoins */
            height: 400px; /* Ajustez la taille selon vos besoins */
            object-fit: cover;
        }
        .container {
            display: flex;
            align-items: center;
        }
        .text-container {
            display: flex;
            flex-direction: column;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # image
    def img_to_bytes(img_path):
        img_bytes = Path(img_path).read_bytes()
        encoded = base64.b64encode(img_bytes).decode()
        return encoded
    
    def img_to_html(img_path):
        img_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
            img_to_bytes(img_path)
        )
        return img_html
    image_url = FOTO_PATH
    
    col_image = st.columns ([3,7])
    with col_image[0]:
        st.markdown("""
        <style>
        .img-fluid {
            max-width: 100%;
            height: auto;
            width: 500px;
            height: 500px;
            margin: 0 auto;
            display: block;
        }
        """, unsafe_allow_html=True)
        st.markdown(img_to_html(image_url), unsafe_allow_html=True)



    with col_image[1]:
        st.text('')
        st.text('')
        st.text('')
        st.text('')
        st.text('')
        st.text('')
        st.text('')
        st.text('')
        st.text('')

        # Conteneur principal
        st.markdown(
            f"""
            <div class="container">
                <div class="text-container">
                    <h1>Welly Oktariana</h1>
                    <h5 style="font-weight:bold; margin-top: 0px; padding-bottom: 0px; padding-top: 0px;">Data Analyst | Data Science | Geospatial Analyst</h4>
                    <p style="style ="text-align: left; font-size:16px; margin-top:0px;">South Tangerang, Banten | 0822-4767-8101 | <a href="mailto:wellyoktariana08@gmail.com">wellyoktariana08@gmail.com</a> | <a href="https://www.linkedin.com/in/wellyoktariana/" target="_blank">LinkedIn</a></p>
                    <div style ="text-align: left; font-size:16px; margin-top:0px;">Data Analyst with 1+ year of experience in building machine learning models, performing spatial analysis, and leveraging data visualization techniques to deliver actionable insights that support strategic decision-making and drive business outcomes.</div>
                    <p align="center">
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
    )
    # Welly Projets
    st.header("Projects")
    # variable contenant la description des projets

    projects = [
        {"image": project1_image, "title": "Ads Click Prediction With Machine Learning", "description": 'Predicts whether an advertisement will be clicked by a user based on a variety of user-specific factors', "link": "https://github.com/wellyokt/portofolio.git", "icon": "https://go-skill-icons.vercel.app/api/icons?i=python,scikitlearn&titles=true"}
    ]
    # Afficher les projets deux par deux
    for i in range(0, len(projects), 2):
        cols = st.columns([1, 0.1, 1])  # Ajout d'une colonne vide pour l'écart
        for j, col in enumerate([cols[0], cols[2]]):  # Utilisation des colonnes 0 et 2 pour les projets
            if i + j < len(projects):
                project = projects[i + j]
                with col:
                    st.markdown(
                        f"""
                        <div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px;'>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"""
                        <style>
                        .project-image {{
                                            max-height: 13px;  /* Définir la hauteur maximale souhaitée */
                                            width: auto;
                                            display: block;
                                            margin-left: auto;
                                            margin-right: auto;
                                        }}
                                        </style>
                                        """,
                                        unsafe_allow_html=True
                        )
                    
                    # Open the image
                    image = Image.open(project["image"])

                    # Display the image
                    st.image(image, use_column_width=True, output_format='png')

                    st.markdown(
                        f"""
                            <h2 style='text-align: center;'>{project['title']}</h2>
                            <p style='text-align: center;'>{project['description']}</p>
                            <p style='text-align: center;'><a href='{project['link']}' target='_blank'>Github</a></p>
                            <p style='text-align: center;'><img src='{project['icon']}'  alt='My Skills'/></p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    

if item =='Predictions':


        # Prediction Form
        with st.form("prediction_form"):
            st.subheader("Enter Parameters")
            
            col1, col2,col3 = st.columns([4.5,0.5,4.5])
            
            with col1:
                bedroom = st.slider(
                    "Bedroom",
                    min_value=0,
                    max_value=10,
                    value=2,
                    help='Total Bedrom'
                )
                
                bathroom = st.slider(
                    "Bathroom",
                    min_value=1,
                    max_value=10,
                    value=3,
                    help="Total Bathroom"
                )
                
                building_area = st.slider(
                    "Building Area (M2)",
                    min_value=10,
                    max_value=1000,
                    value=350,
                    step=50,
                    help="Building Area (M2)"
                )
            
            with col3:
                land_area = st.slider(
                    "Land Area (M2)",
                    min_value=10,
                    max_value=1000,
                    value=250,
                    step=50,
                    help="Land Area (M2)"
                )
       
                # Corrected Latitude Input
                lat = st.number_input(
                    "Latitude",
                    min_value=-90.0,  # Correct latitude range
                    max_value=90.0,
                    value=-7.3137835,  # Default value
                    step=0.0000001,
                    format ="%.7f",
                    help="Coordinate Latitude (-90 to 90)"
                )
                long = st.number_input(
                    "Longitude",
                    min_value=-180.0,  # Correct longitude range
                    max_value=180.0,
                    value= 112.6684179,
                    step=0.0000001,
                    format ="%.7f",
                    help="Coordinate Longitude (-180 to 180)"
                )
    
  
      

            st.text('')
            st.text('')
            submitted = st.form_submit_button("Gerate Price Prediction", type='primary',use_container_width=True)

        if submitted:
            input_data = {
                "bedroom":bedroom,
                "bathroom": bathroom,
                "building_area": building_area,
                "land_area": land_area,
                "latitude": lat,
                "longitude": long}
            df = pd.DataFrame(input_data, index=[0])
            df['geometry'] =df.apply(lambda x:Point(x['longitude'],x['latitude']),axis=1)
            df= gpd.GeoDataFrame(df, geometry='geometry', crs=4326)
                        # try:
            with st.spinner('Making prediction...'):
                try:
                    response = requests.post(
                        "http://localhost:8000/predict",
                        json=input_data
                    )
                    
                    
                    if response.status_code == 200:
                        prediction = response.json()
                                 

                except:
                    model_path = os.path.join(project_root, 'artifacts', 'model', 'model.pkl')
                    scaler_path = os.path.join(project_root, 'artifacts', 'model', 'scaler.pkl')
                    dl = DataLoader(model_path, scaler_path)

                    model = dl.load_model()
                    scaler = dl.load_scaler()

                    data = df
                    data_processor = Preprocessor(data,scaler)
                    x_img, x_tx, geom_buffer= data_processor.extract_feature()
                
                    prediction = model.predict([x_img,x_tx]) # Adjust indexing based on your model's output
                    prediction =int(prediction[0][0])

    
                    with st.container(border=True):

                        st.markdown('Map View')
                        st.markdown("""
                            <style>
                            iframe {
                                width: 100%;
                                height: 100%:
                            }
                            </style>
                            """, unsafe_allow_html=True)
                        m = folium.Map(location=[df['latitude'].values[0], df['longitude'].values[0]], zoom_start=14, tiles='Esri.WorldImagery')

                        df.to_crs(3395).buffer(1000, cap_style='square').to_crs(4326).explore(m=m,color='grey')
                        # Add GeoDataFrame Points to Map
                        for _, row in df.iterrows():
                            folium.Marker(
                                location=[row.geometry.y, row.geometry.x],  # Latitude, Longitude
                                popup=f"Location: {row.geometry.y}, {row.geometry.x}",
                                icon=folium.Icon(color="blue", icon="info-sign")
                            ).add_to(m)

                        folium_static(m)
                    
                    geom_buffer = df.to_crs(3395).buffer(1000, cap_style='square').to_crs(4326)
                    bbox =list(geom_buffer[0] .bounds)
                    # Get satellite image (in-memory, no saving)

                    st.success(f"#### House Price Prediction: Rp{prediction}")



                
                    


    