#app.py
import streamlit as st
import pandas as pd
import numpy as np
import time
from keras.models import load_model
import requests
import pickle
import openai
import os
import urllib.request
from datetime import datetime, timedelta
from queue import Queue
from generator_script import generate_continuous_data
from model_utils import detect_anomalies, generate_diagnosis_and_recommendation, generate_prompts_from_anomalies, inverse_transform, create_sequences, load_model_from_github, send_email
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Paths to files
generator_path = 'https://github.com/tadiwamark/pdM_Genset_Analytics/releases/download/gan/generator_model.h5'
discriminator_path = 'https://github.com/tadiwamark/pdM_Genset_Analytics/releases/download/gan/discriminator_model.h5'
optimizer = 'adam'
generator_loss = 'binary_crossentropy'
discriminator_loss = 'binary_crossentropy'

# Load Model
generator_model = load_model_from_github(generator_path)
discriminator_model = load_model_from_github(discriminator_path)
generator_model.compile(optimizer=optimizer, loss=generator_loss)
discriminator_model.compile(optimizer=optimizer, loss=discriminator_loss)



def main():
    st.title('FG Wilson Generator Monitoring Dashboard')


    if not st.session_state.get('api_key'):
        st.session_state.api_key = st.sidebar.text_input("Enter your OpenAI API Key:")
        if st.session_state.api_key:
            openai.api_key = st.session_state.api_key

    st.sidebar.title('Generator Controls')
    generator_state = st.sidebar.button('Start/Stop Generator')

    if 'generator_on' not in st.session_state:
        st.session_state['generator_on'] = False

    if generator_state:
        st.session_state['generator_on'] = not st.session_state['generator_on']

    data_placeholder = st.empty()
    insights_placeholder = st.empty()
    graph_placeholder1 = st.empty()
    graph_placeholder2 = st.empty()
    graph_placeholder3 = st.empty()
    graph_placeholder4 = st.empty()
    status_placeholder = st.empty()
    anomaly_detection_placeholder = st.empty()

    if 'anomaly_queue' not in st.session_state:
        st.session_state.anomaly_queue = Queue()


    if st.session_state['generator_on']:
        start_time = datetime.now()
        data_generator = generate_continuous_data(start_time)
        simulated_data_df = pd.DataFrame()
        accumulated_data = []
        anomalies_timestamps = []


        while st.session_state['generator_on']:
            try:
                new_data = next(data_generator)
                accumulated_data.append(new_data)
                simulated_data_df = pd.concat(accumulated_data).reset_index(drop=True)

                if not simulated_data_df.empty:
                    numeric_column_names = simulated_data_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    simulated_data_df['Load_Factor'] = simulated_data_df['AverageCurrent(A)'] / simulated_data_df['Phase1Current(A)'].max()
                    simulated_data_df['Temp_Gradient'] = simulated_data_df['ExhaustTemp(°C)'] - simulated_data_df['CoolantTemp( °C)']
                    simulated_data_df['Pressure_Ratio'] = simulated_data_df['inLetPressure(KPa)'] / simulated_data_df['outLetPressure(KPa)']
                    simulated_data_df['Imbalance_Current'] = simulated_data_df[['Phase1Current(A)', 'Phase2Current(A)', 'Phase3Current(A)']].std(axis=1)
                    simulated_data_df['Power_Factor_Deviation'] = 1 - simulated_data_df['PowerFactor'].abs()
                    domain_features = ['Load_Factor', 'Temp_Gradient', 'Pressure_Ratio', 'Imbalance_Current','Power_Factor_Deviation']
                    numeric_column_names += domain_features

                    numeric_columns = simulated_data_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    numeric_columns += domain_features
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(simulated_data_df[numeric_columns])
                    scaled_data_df = pd.DataFrame(scaled_data, columns=numeric_columns)
                    scaled_data_seq = create_sequences(scaled_data_df, 10)

                    fig1, ax1 = plt.subplots(figsize=(15, 8))
                    ax1.plot(simulated_data_df['Time'], simulated_data_df['AverageCurrent(A)'].rolling(window=10).mean(), label='Average Current (A)', color='blue')
                    ax1.plot(simulated_data_df['Time'], simulated_data_df['Phase1Current(A)'].rolling(window=10).mean(), label='Phase 1 Current (A)', color='red', linestyle='--')
                    ax1.plot(simulated_data_df['Time'], simulated_data_df['Phase2Current(A)'].rolling(window=10).mean(), label='Phase 2 Current (A)', color='green', linestyle='--')
                    ax1.plot(simulated_data_df['Time'], simulated_data_df['Phase3Current(A)'].rolling(window=10).mean(), label='Phase 3 Current (A)', color='purple', linestyle='--')
                    ax1.set_xlabel('Time')
                    ax1.set_ylabel('Current (A)')
                    ax1.legend()

                    for anomaly_time in anomalies_timestamps:
                        ax1.axvline(anomaly_time, color='red', linestyle='--')
                    
                    graph_placeholder1.pyplot(fig1)

                    

                    fig2, ax2 = plt.subplots(figsize=(15, 8))
                    ax2.plot(simulated_data_df['Time'], simulated_data_df['ExhaustTemp(°C)'].rolling(window=10).mean(), label='Exhaust Temp (°C)', color='blue')
                    ax2.plot(simulated_data_df['Time'], simulated_data_df['CoolantTemp( °C)'].rolling(window=10).mean(), label='Coolant Temp (°C)', color='red', linestyle='--')
                    ax2.set_xlabel('Time')
                    ax2.set_ylabel('Temperature (°C)')
                    ax2.legend()

                    for anomaly_time in anomalies_timestamps:
                        ax2.axvline(anomaly_time, color='red', linestyle='--')
                    
                    graph_placeholder2.pyplot(fig2)

                    

                    fig3, ax3 = plt.subplots(figsize=(15, 8))
                    ax3.plot(simulated_data_df['Time'], simulated_data_df['inLetPressure(KPa)'].rolling(window=10).mean(), label='Inlet Pressure (KPa)', color='blue')
                    ax3.plot(simulated_data_df['Time'], simulated_data_df['outLetPressure(KPa)'].rolling(window=10).mean(), label='Outlet Pressure (KPa)', color='red', linestyle='--')
                    ax3.set_xlabel('Time')
                    ax3.set_ylabel('Pressure (KPa)')
                    ax3.legend()

                    for anomaly_time in anomalies_timestamps:
                        ax3.axvline(anomaly_time, color='red', linestyle='--')

                    
                    graph_placeholder3.pyplot(fig3)

                    data_placeholder.dataframe(simulated_data_df)

                    plt.close(fig1)
                    plt.close(fig2)
                    plt.close(fig3)



                    if len(accumulated_data) >= 15:  # Process data every 60 records (5 minutes assuming 5 sec interval)
                        optimal_threshold = 0.7
                        features = scaled_data.shape[1]
                        anomalies, real_predictions, fake_predictions = detect_anomalies(generator_model, discriminator_model, scaled_data_seq, features)
                        real_predictions = discriminator_model.predict(scaled_data_seq)
                        
                        anomalies_indices = np.where(real_predictions < optimal_threshold)[0]
                        anomalies = scaled_data_seq[anomalies_indices]
                        print(f"Detected {len(anomalies)} potential anomalies.")
                        anomalies_data = inverse_transform(anomalies.reshape(-1, features), scaler)
                        anomalies_df = pd.DataFrame(anomalies_data, columns=numeric_columns)
                        
                        anomaly_data = generate_prompts_from_anomalies(anomalies_df)
                        
                        for prompt in anomaly_data:
                            st.session_state.anomaly_queue.put(prompt)
                            
                        for idx in anomalies_indices:
                            anomalies_timestamps.append(simulated_data_df['Time'].iloc[idx])


                        # Display success message if anomaly detection model has run
                        anomaly_detection_placeholder.success("Anomaly detection model has run successfully and prompts have been stored in the queue.")

                        # Display insights from queue at regular intervals
                        if not st.session_state.anomaly_queue.empty():
                            prompt = st.session_state.anomaly_queue.get()
                            if prompt:
                                diagnosis = generate_diagnosis_and_recommendation(prompt)
                                if diagnosis:
                                    insights_placeholder.markdown(f"## Insights\n- **Model Diagnosis and Recommendation:**\n{diagnosis}")
                                    send_email("Generator Anomaly Alert", diagnosis)
                                else:
                                    insights_placeholder.markdown(f"## Insights\n- **Model Diagnosis and Recommendation:**\nNo recommendations available.")
                            else:
                                insights_placeholder.markdown(f"## Insights\n- **Model Diagnosis and Recommendation:**\nNo prompts generated.")
                        
    
                        # Reset index for new batch, keep last 60 records for continuity
                        simulated_data_df = simulated_data_df.iloc[-60:].reset_index(drop=True)
                        accumulated_data = [simulated_data_df]

                
                
                time.sleep(1)

            except StopIteration:
                break

        status_placeholder.success("Generator is currently ON.")

    else:
        status_placeholder.warning("Generator is currently OFF. Use the sidebar to start the generator.")

if __name__ == "__main__":
    main()
