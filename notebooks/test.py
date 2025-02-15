import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import geopandas as gpd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.layers import Input, Dense, Concatenate, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from stn import spatial_transformer_network as STN

transaction_data=gpd.read_parquet(r"D:\ML5\final_projectt\house_price.parquet")
transaction_data["image_filename"]=transaction_data["id"].apply(lambda x:f"{x}.png")

# Calculate Q1, Q3, and IQR
Q1 = np.percentile(transaction_data['price'].values.tolist(), 25)
Q3 = np.percentile(transaction_data['price'].values.tolist(), 75)
IQR = Q3 - Q1

# Define bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

transaction_data = transaction_data[(transaction_data['price']<=upper_bound)&(transaction_data['price']>=lower_bound)]


image_folder = r"D:\ML5\final_projectt\artifacts\png"



# Function to load and preprocess images
def load_images(image_filenames, img_size=(224, 224)):
    images = []
    matched_filenames = []

    for filename in image_filenames:
        img_path = os.path.join(image_folder, filename)
        if os.path.exists(img_path):  # Ensure the image file exists
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            img = img / 255.0  # Normalize
            images.append(img)
            matched_filenames.append(filename)
    

    return np.array(images), matched_filenames

# Get the list of filenames from the dataset
image_filenames = transaction_data["image_filename"].tolist()

# Load only images that exist in the dataset
satellite_images, valid_filenames = load_images(image_filenames)

print(f"Loaded {len(satellite_images)} images out of {len(image_filenames)} transaction records.")

# Keep only transaction records where the image exists
filtered_data = transaction_data[transaction_data["image_filename"].isin(valid_filenames)].reset_index(drop=True)
# filtered_data['price'] = filtered_data['price'].apply(np.log)


print(f"Filtered dataset size: {filtered_data.shape}")

selected_features = ['bedroom', 'bathroom', 'building_area', 'land_area']
X_transaction = filtered_data[selected_features].fillna(0).values # Convert to NumPy array
y_prices = filtered_data["price"].values.reshape(-1, 1)  

# # Standardize transaction data
scaler = StandardScaler()
X_transaction = scaler.fit_transform(X_transaction)

print(f"Transaction features shape: {X_transaction.shape}, Labels shape: {y_prices.shape}")

from sklearn.model_selection import train_test_split

# Split into training (80%) and testing (20%)
X_train_img, X_test_img, X_train_tx, X_test_tx, y_train, y_test = train_test_split(
    satellite_images, X_transaction, y_prices, test_size=0.2, random_state=42
)

print(f"Training set: {X_train_img.shape}, {X_train_tx.shape}, {y_train.shape}")
print(f"Testing set: {X_test_img.shape}, {X_test_tx.shape}, {y_test.shape}")


from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Concatenate, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam  # ✅ Import Adam optimizer


# Define input shapes
input_shape_img = (224, 224, 3)  # Image input shape
input_shape_tx = (X_train_tx.shape[1],)  # Transaction data input shape

# Image Input (ResNet Feature Extraction)
image_input = Input(shape=input_shape_img, name="Image_Input")

# Initialize ResNet50 without the top layer
# resnet_base = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape_img)
vgg16 =VGG16(weights="imagenet", include_top=False, input_shape=input_shape_img)
vgg16.trainable = False  # Freeze ResNet50 layers during initial training

# Extract features using ResNet50
resnet_features = vgg16(image_input)
resnet_output = GlobalAveragePooling2D()(resnet_features)
resnet_output = Dense(256, activation='relu')(resnet_output)
resnet_output = Dense(128, activation='relu')(resnet_output)
resnet_output = Dense(64, activation='relu')(resnet_output) 
resnet_output = Dense(32, activation='relu')(resnet_output) 
resnet_output = Dense(10, activation='relu')(resnet_output) 
# resnet_output = Dense(1, activation='relu')(resnet_output) 


#  Transaction Data Input
transaction_input = Input(shape=input_shape_tx, name="Transaction_Input")

#  Merge Image Features & Transaction Data
merged_features = Concatenate()([resnet_output, transaction_input])

# Fully Connected Layers for Regression
x = Dense(128, activation='relu')(merged_features)
x = Dropout(0.2)(x)  # Adding dropout with 30% dropout rate
x = Dense(64, activation='relu')(x)
x = Dropout(0.1)(x)  # Adding dropout with 20% dropout rate
x = Dense(32, activation='relu')(x)
output = Dense(1, activation='linear', name="Price_Output")(x)  # Ensure this is a Keras Tensor

# Define the final model
model = Model(inputs=[image_input, transaction_input], outputs=output)

model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])


# Model Summary
model.summary()

# Train the model using the matched dataset
model.fit(
    [X_train_img, X_train_tx], y_train, 
    validation_data=([X_test_img, X_test_tx], y_test),
    epochs=2, batch_size=50
)