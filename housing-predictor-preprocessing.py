import pandas as pd
import numpy as np

def generate_synthetic_housing_data(num_samples=1000):
    """
    Generate synthetic housing data for Anchorage
    
    Parameters:
    -----------
    num_samples : int, default 1000
        Number of synthetic data points to generate
    
    Returns:
    --------
    pandas.DataFrame
        Synthetic housing dataset
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Neighborhoods with their base price multipliers
    neighborhoods = {
        'Downtown': 1.2,
        'Midtown': 1.1,
        'South Anchorage': 1.0,
        'East Anchorage': 0.9,
        'Eagle River': 0.8
    }
    
    # Generate data
    data = {
        'square_feet': np.random.normal(1500, 500, num_samples).clip(500, 5000),
        'bedrooms': np.random.randint(1, 6, num_samples),
        'bathrooms': np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0], num_samples),
        'year_built': np.random.randint(1950, 2024, num_samples),
        'neighborhood': np.random.choice(list(neighborhoods.keys()), num_samples)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Calculate base price
    base_price = 250000
    
    # Calculate price with some randomness and neighborhood multipliers
    df['price'] = (
        base_price + 
        (df['square_feet'] - 1500) * 150 +  # Price per square foot
        (df['bedrooms'] - 3) * 50000 +      # Bedroom premium
        (df['bathrooms'] - 2) * 25000 -     # Bathroom adjustment
        (2024 - df['year_built']) * 1000    # Depreciation
    )
    
    # Apply neighborhood multipliers
    df['price'] *= df['neighborhood'].map({
        'Downtown': 1.2,
        'Midtown': 1.1,
        'South Anchorage': 1.0,
        'East Anchorage': 0.9,
        'Eagle River': 0.8
    })
    
    # Add some random noise
    df['price'] *= np.random.normal(1, 0.1, num_samples)
    
    return df

def load_and_preprocess_data():
    """
    Load and preprocess housing data
    
    Returns:
    --------
    pandas.DataFrame
        Preprocessed housing dataset
    """
    # Generate synthetic data
    df = generate_synthetic_housing_data()
    
    # Encode neighborhood
    neighborhood_map = {
        'Downtown': 0,
        'Midtown': 1,
        'South Anchorage': 2,
        'East Anchorage': 3,
        'Eagle River': 4
    }
    df['neighborhood_encoded'] = df['neighborhood'].map(neighborhood_map)
    
    return df
