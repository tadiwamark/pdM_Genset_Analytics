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
import requests
import pickle
import openai
import os
import urllib.request
from datetime import datetime, timedelta
from generator_script import generate_continuous_data
from model_utils import detect_anomalies, generate_diagnosis_and_recommendation, generate_prompts_from_anomalies, inverse_transform, create_sequences, load_model_from_github, download_and_load_scaler
from sklearn.preprocessing import StandardScaler




# Paths to files
scaler_path = 'https://github.com/tadiwamark/pdM_Genset_Analytics/releases/download/gan/scaler.gz'
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
      end_time = start_time + timedelta(hours=3)
      # Simulate data generation for 3 hours
      simulated_data_df = generate_continuous_data(start_time, end_time)

      numeric_columns = simulated_data_df.select_dtypes(include=[np.number])

      if not numeric_columns.empty:
          numeric_columns = numeric_columns.astype(float).fillna(numeric_columns.mean())
          # Merge back the numeric columns to the main DataFrame
          for column in numeric_columns:
              simulated_data_df[column] = numeric_columns[column]


      

      if simulated_data_df.empty:
          print("The DataFrame is empty.")
      else:
          print("The DataFrame has data.")

      simulated_data_df
      simulated_data_df['Load_Factor'] = simulated_data_df['AverageCurrent(A)'] / simulated_data_df['Phase1Current(A)'].max()
      simulated_data_df['Temp_Gradient'] = simulated_data_df['ExhaustTemp(°C)'] - simulated_data_df['CoolantTemp( °C)']
      simulated_data_df['Pressure_Ratio'] = simulated_data_df['inLetPressure(KPa)'] / simulated_data_df['outLetPressure(KPa)']
      simulated_data_df['Imbalance_Current'] = simulated_data_df[['Phase1Current(A)', 'Phase2Current(A)', 'Phase3Current(A)']].std(axis=1)
      simulated_data_df['Power_Factor_Deviation'] = 1 - simulated_data_df['PowerFactor'].abs()






      numerical_features = simulated_data_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
      domain_features = ['Load_Factor', 'Temp_Gradient', 'Pressure_Ratio', 'Imbalance_Current','Power_Factor_Deviation']
      numerical_features += domain_features

      # Normalize the data
      scaler = StandardScaler()
      scaled_data = scaler.fit_transform(simulated_data_df[numerical_features])

      # Double-check if scaled_data contains any NaN values
      assert not np.isnan(scaled_data).any(), "Scaled data contains NaN after re-scaling"
      assert not np.isinf(scaled_data).any(), "Scaled data contains Inf after re-scaling"

      # Convert scaled data to DataFrame
      scaled_data_df = pd.DataFrame(scaled_data, columns=numerical_features)

      

      sequence_length = 10

      scaled_data_seq = create_sequences(scaled_data_df, sequence_length)
      required_features = scaled_data_df.columns[:25]  # This now works because scaled_data_df is a DataFrame
      scaled_data_seq = scaled_data_df[required_features].values

      for _, row in simulated_data_df.iterrows():
          # Display simulated data
          data_placeholder.dataframe(row.to_frame().T)

          # Detect anomalies in the simulated data

          optimal_threshold = 0.7

          features = scaled_data.shape[1]

          anomalies, real_predictions, fake_predictions = detect_anomalies(generator, discriminator, real_data, numerical_features, features, threshold=0.5)

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
