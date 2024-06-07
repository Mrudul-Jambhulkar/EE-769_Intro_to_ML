#ref https://chat.openai.com/share/93edae38-3a03-4720-8310-d593f258d051 and https://www.youtube.com/watch?v=Q1yL5LmtMbo
import pickle
import pandas as pd
import streamlit as st
import numpy as np

# Load the trained model
model = pickle.load(open('trained_model.sav', 'rb'))

# Read red wine dataset
df_red = pd.read_csv('winequality-red.csv')

# Split the single column into multiple columns
df_red = df_red.iloc[:, 0].str.split(";", expand=True)

# Convert strings to numbers
df_red = df_red.apply(pd.to_numeric, errors='coerce')

# Renaming the columns
df_red.columns = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality"
]
st.title('Quality prediction of Red Wine')
st.image('redwine.jpeg', caption='Red Wine', use_column_width=True , width=150)


st.sidebar.header('Wine Features')

# Define the sliders for various features
feature_sliders = {}
for feature in df_red.columns[:-1]:  # Exclude the target column 'quality'
    feature_sliders[feature] = st.sidebar.slider(f'{feature.capitalize()}',
                                                 min_value=df_red[feature].min(),
                                                 max_value=df_red[feature].max(),
                                                 value=df_red[feature].mean(),
                                                 step=0.01)

# Add a button to trigger model prediction
if st.button('Predict'):
    # Prepare the input features as a numpy array
    input_features = np.array([[feature_sliders[feature] for feature in df_red.columns[:-1]]])
    
    # Perform prediction using the loaded model
    prediction = model.predict(input_features)
    
    # Display the prediction result
    st.write('Quality of wine :', prediction)  # Assuming the prediction is a single value
