# app.py

import streamlit as st
import pandas as pd
import numpy as np
import time
from keras.models import load_model
import requests
import pickle
import openai
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from generator_script import generate_continuous_data
from model_utils import detect_anomalies, generate_diagnosis_and_recommendation, generate_prompts_from_anomalies, inverse_transform, create_sequences, load_model_from_github
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Email alert function
def send_email(subject, body):
    sender_email = "youremail@example.com"
    receiver_email = "receiver@example.com"
    password = "yourpassword"

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    message.attach(MIMEText(body, "plain"))

    with smtplib.SMTP_SSL("smtp.example.com", 465) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())

# Paths to files
generator_path = 'https://github.com/tadiwamark/pdM_Genset_Analytics/releases/download/gan/generator_model.h5'
discriminator_path = 'https://github.com/tadiwamark/pdM_Genset_Analytics/releases/download/gan/discriminator_model.h5'

optimizer = 'adam'
generator_loss = 'binary_crossentropy'
discriminator_loss = 'binary_crossentropy'

# Load Model
generator = load_model_from_github(generator_path)
discriminator = load_model_from_github(discriminator_path)

generator.compile(optimizer=optimizer, loss=generator_loss)
discriminator.compile(optimizer=optimizer, loss=discriminator_loss)

def main():
    # Streamlit App UI
    st.set_page_config(page_title="FG Wilson Generator Monitoring Dashboard", layout="wide")
    st.title('FG Wilson Generator Monitoring Dashboard')

    # Get API Key for GPT-3.5
    if not st.session_state.get('api_key'):
        st.session_state.api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
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
    graph_placeholder = st.empty()
    anomalies_placeholder = st.empty()

    if st.session_state['generator_on']:
        data_generator = generate_continuous_data()

        for simulated_data_df in data_generator:
            if not simulated_data_df.empty:
                # Prepare data for anomaly detection
                numeric_column_names = simulated_data_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                # Feature engineering
                simulated_data_df['Load_Factor'] = simulated_data_df['AverageCurrent(A)'] / simulated_data_df['Phase1Current(A)'].max()
                simulated_data_df['Temp_Gradient'] = simulated_data_df['ExhaustTemp(°C)'] - simulated_data_df['CoolantTemp( °C)']
                simulated_data_df['Pressure_Ratio'] = simulated_data_df['inLetPressure(KPa)'] / simulated_data_df['outLetPressure(KPa)']
                simulated_data_df['Imbalance_Current'] = simulated_data_df[['Phase1Current(A)', 'Phase2Current(A)', 'Phase3Current(A)']].std(axis=1)
                simulated_data_df['Power_Factor_Deviation'] = 1 - simulated_data_df['PowerFactor'].abs()

                domain_features = ['Load_Factor', 'Temp_Gradient', 'Pressure_Ratio', 'Imbalance_Current', 'Power_Factor_Deviation']
                numeric_column_names += domain_features

                # Normalize and prepare sequences
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(simulated_data_df[numeric_column_names])
                scaled_data_df = pd.DataFrame(scaled_data, columns=numeric_column_names)

                scaled_data_seq = create_sequences(scaled_data_df, 10)

                # Graphical display
                fig, ax = plt.subplots()
                ax.plot(simulated_data_df.index, simulated_data_df['AverageCurrent(A)'], label='Average Current (A)', color='blue')
                ax.plot(simulated_data_df.index, simulated_data_df['Phase1Current(A)'], label='Phase 1 Current (A)', color='red', linestyle='--')
                ax.plot(simulated_data_df.index, simulated_data_df['Phase2Current(A)'], label='Phase 2 Current (A)', color='green', linestyle='--')
                ax.plot(simulated_data_df.index, simulated_data_df['Phase3Current(A)'], label='Phase 3 Current (A)', color='purple', linestyle='--')
                ax.set_xlabel('Time')
                ax.set_ylabel('Current (A)')
                ax.legend()
                graph_placeholder.pyplot(fig)

                for _, row in simulated_data_df.iterrows():
                    # Display simulated data
                    

                    # Detect anomalies in the simulated data
                    optimal_threshold = 0.7
                    features = scaled_data.shape[1]
                    anomalies, real_predictions, fake_predictions = detect_anomalies(generator, discriminator, scaled_data_seq, numeric_column_names)
                    real_predictions = discriminator.predict(scaled_data_seq)
                    anomalies_indices = np.where(real_predictions < optimal_threshold)[0]
                    anomalies = scaled_data_seq[anomalies_indices]

                    # Identify characteristics of anomalies
                    anomalies_data = inverse_transform(anomalies.reshape(-1, features), scaler)
                    anomalies_df = pd.DataFrame(anomalies_data, columns=numeric_column_names)

                    # Display anomalous data
                    anomalies_placeholder.dataframe(anomalies_df.drop(columns=['Anomaly_Type']))

                    # Generate prompts for each anomaly
                    anomaly_data = generate_prompts_from_anomalies(anomalies_df)

                    for prompt in anomaly_data:
                        diagnosis = generate_diagnosis_and_recommendation(prompt)
                        insights_placeholder.markdown(f"## Insights\n- **Model Diagnosis and Recommendation:**\n{diagnosis}")

                        # Send email alert
                        send_email("Generator Anomaly Alert", diagnosis)

                        time.sleep(60)

    else:
        st.write("Generator is currently OFF. Use the sidebar to start the generator.")

if __name__ == "__main__":
    main()
