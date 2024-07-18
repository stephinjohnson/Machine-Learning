import pandas as pd
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define a sample input for prediction (replace with actual values)
sample_input = {
    'longitude': -122.23,
    'latitude': 37.88,
    'housing_median_age': 41,
    'total_rooms': 880,
    'total_bedrooms': 129,
    'population': 400,
    'households': 126,
    'median_income': 8.3252,
    'ocean_proximity_<1H OCEAN': 0,
    'ocean_proximity_INLAND': 0,
    'ocean_proximity_ISLAND': 0,
    'ocean_proximity_NEAR BAY': 1,
    'ocean_proximity_NEAR OCEAN': 0
}

# Convert the sample input to a DataFrame
sample_df = pd.DataFrame([sample_input])

# Scale the sample input
sample_scaled = scaler.transform(sample_df)

# Make a prediction
prediction = model.predict(sample_scaled)

print(f'Predicted House Price: ${prediction[0]:,.2f}')
