import streamlit as st
import numpy as np
import joblib
import os

# Load the trained model
model_path = os.path.join('model', 'predictive_model.pkl')
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("Model file not found. Please ensure the model is trained and saved.")

# Set up the Streamlit app
st.title('Predictive Maintenance Model')

st.write("""
### Enter Sensor Data to Predict Maintenance
""")

# Input fields for sensor data
sensor1 = st.number_input('Sensor 1', min_value=0.0, max_value=1.0, value=0.5)
sensor2 = st.number_input('Sensor 2', min_value=0.0, max_value=1.0, value=0.5)
sensor3 = st.number_input('Sensor 3', min_value=0.0, max_value=1.0, value=0.5)
sensor4 = st.number_input('Sensor 4', min_value=0.0, max_value=1.0, value=0.5)

# Prediction button
if st.button('Predict Maintenance'):
    # Ensure that input data is in correct shape
    input_data = np.array([[sensor1, sensor2, sensor3, sensor4]])
    prediction = model.predict(input_data)
    
    # Display prediction result
    if prediction[0] == 1:
        st.error('Maintenance is needed!')
    else:
        st.success('No maintenance required.')

# Footer
st.write('Predictive Maintenance Model by ABHI')
