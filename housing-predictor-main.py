import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Import custom modules
from data_preprocessing import load_and_preprocess_data
from model_training import train_model

def load_or_train_model():
    """
    Load existing model or train a new one if not available
    """
    model_path = 'anchorage_housing_model.joblib'
    
    if os.path.exists(model_path):
        # Load existing model
        model = joblib.load(model_path)
        return model
    else:
        # Train a new model
        df = load_and_preprocess_data()
        model = train_model(df)
        
        # Save the model
        joblib.dump(model, model_path)
        return model

def predict_house_price(model, features):
    """
    Predict house price using the trained model
    """
    # Prepare input features
    input_features = np.array([
        features['square_feet'],
        features['bedrooms'],
        features['bathrooms'],
        features['year_built'],
        features['neighborhood_encoded']
    ]).reshape(1, -1)
    
    # Make prediction
    predicted_price = model.predict(input_features)[0]
    return predicted_price

def main():
    st.title('Anchorage Housing Price Predictor')
    
    # Load or train the model
    model = load_or_train_model()
    
    # Sidebar for input features
    st.sidebar.header('House Features')
    
    # Input features
    square_feet = st.sidebar.number_input(
        'Square Feet', 
        min_value=500, 
        max_value=5000, 
        value=1500
    )
    
    bedrooms = st.sidebar.number_input(
        'Number of Bedrooms', 
        min_value=1, 
        max_value=10, 
        value=3
    )
    
    bathrooms = st.sidebar.number_input(
        'Number of Bathrooms', 
        min_value=1.0, 
        max_value=6.0, 
        value=2.0, 
        step=0.5
    )
    
    year_built = st.sidebar.number_input(
        'Year Built', 
        min_value=1900, 
        max_value=2024, 
        value=2000
    )
    
    # Neighborhood selection with predefined encoding
    neighborhoods = {
        'Downtown': 0,
        'Midtown': 1,
        'South Anchorage': 2,
        'East Anchorage': 3,
        'Eagle River': 4
    }
    neighborhood = st.sidebar.selectbox(
        'Neighborhood', 
        list(neighborhoods.keys()), 
        index=2
    )
    neighborhood_encoded = neighborhoods[neighborhood]
    
    # Predict button
    if st.sidebar.button('Predict House Price'):
        # Prepare features dictionary
        features = {
            'square_feet': square_feet,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'year_built': year_built,
            'neighborhood_encoded': neighborhood_encoded
        }
        
        # Make prediction
        predicted_price = predict_house_price(model, features)
        
        # Display prediction
        st.success(f'Estimated House Price: ${predicted_price:,.2f}')
        
        # Additional insights
        st.subheader('Prediction Insights')
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric('Square Feet', f'{square_feet} sq ft')
            st.metric('Bedrooms', bedrooms)
        
        with col2:
            st.metric('Bathrooms', bathrooms)
            st.metric('Neighborhood', neighborhood)

    # Model Performance Section
    st.sidebar.header('Model Performance')
    st.sidebar.write('Model: Random Forest Regressor')
    st.sidebar.write('Training Data: Simulated Anchorage Housing Dataset')

if __name__ == '__main__':
    main()
