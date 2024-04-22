# -*- coding: utf-8 -*-
"""app.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1VsF7C9ooqnn_5in6JQGffvnkPouEG-B0
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import gzip
from keras.models import load_model
import pickle
import openai
import os
import urllib.request
from datetime import datetime
from generator_script import generate_continuous_data
from model_utils import detect_anomalies, generate_diagnosis_and_recommendation, generate_prompts_from_anomalies, inverse_transform, create_sequences, load_model_from_github, download_and_load_scaler
from sklearn.preprocessing import StandardScaler




# Paths to files
scaler_path = '/content/drive/My Drive/Colab Notebooks/Final_Year_Project/scaler.gz'
generator_path = 'https://github.com/tadiwamark/pdM_Genset_Analytics/releases/download/gan/generator_model.h5'
discriminator_path = 'https://github.com/tadiwamark/pdM_Genset_Analytics/releases/download/gan/discriminator_model.h5'



# Load Model
generator = load_model_from_github(generator_path)
discriminator = load_model_from_github(discriminator_path)




def main():
  # Streamlit App UI
  st.title('FG Wilson Generator Monitoring Dashboard')

  # Get API Key for GPT-3.5
  if not st.session_state.get('api_key'):
    st.session_state.api_key = st.sidebar.text_input("Enter your OpenAI API Key:")
    if st.session_state.api_key:
        openai.api_key = st.session_state.api_key


  # Sidebar controls for generator operation and model uploads
  st.sidebar.title('Generator Controls')
  generator_state = st.sidebar.button('Start/Stop Generator')

  # Session states for generator operation and data generation
  if 'generator_on' not in st.session_state:
      st.session_state['generator_on'] = False

  # Start/Stop generator
  if generator_state:
      st.session_state['generator_on'] = not st.session_state['generator_on']


  # Main dashboard for displaying generator data and insights
  data_placeholder = st.empty()
  insights_placeholder = st.empty()

  if st.session_state['generator_on']:
      start_time = datetime.now()
      # Simulate data generation for 3 hours
      simulated_data_df = generate_continuous_data(start_time, 3)

      simulated_data_df = simulated_data_df.astype(float).fillna(simulated_data_df.mean())

      if simulated_data_df.empty:
          print("The DataFrame is empty.")
      else:
          print("The DataFrame has data.")






      numerical_features = simulated_data_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
      domain_features = ['Load_Factor', 'Temp_Gradient', 'Pressure_Ratio', 'Imbalance_Current','Power_Factor_Deviation']
      numerical_features += domain_features

      # Normalize the data
      # Create an instance of StandardScaler
      scaler = StandardScaler()

      # Fit and transform the data
      scaled_data = scaler.fit_transform(simulated_data_df)

      sequence_length = 10

      scaled_data_seq = create_sequences(scaled_data, sequence_length)

      for _, row in simulated_data_df.iterrows():
          # Display simulated data
          data_placeholder.dataframe(row.to_frame().T)

          # Detect anomalies in the simulated data

          optimal_threshold = 0.7

          anomalies, real_predictions, fake_predictions = detect_anomalies(generator, discriminator, scaled_data_seq, threshold=0.5)

          real_predictions = discriminator.predict(scaled_data_seq)



          anomalies_indices = np.where(real_predictions < optimal_threshold)[0]
          anomalies = scaled_data_seq[anomalies_indices]


          # Identify characteristics of anomalies
          anomalies_data = inverse_transform(anomalies.reshape(-1, features), scaler)

          # Convert anomalies_data back to a DataFrame for easier analysis
          anomalies_df = pd.DataFrame(anomalies_data, columns=numerical_features)


          # Generate prompts for each anomaly
          anomaly_data = generate_prompts_from_anomalies(anomalies_df)


          # Get interpretation and recommendation for the detected anomaly
          diagnosis = generate_diagnosis_and_recommendation(anomaly_data)
          print("Model Diagnosis and Recommendation:")
          print(diagnosis)

          # Display interpretation and recommendation
          insights_placeholder.markdown(f"""
          ## Insights
          - **Model Diagnosis and Recommendation:\n** {diagnosis}
          """)

          # Wait before updating with new data
          time.sleep(10)  # Adjust the sleep time as needed for your simulation speed

  else:
      st.write("Generator is currently OFF. Use the sidebar to start the generator.")



if __name__ == "__main__":
    main()
