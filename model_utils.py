#model_utils.py
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import urllib.request
from transformer_encoder_block import TransformerEncoderBlock
import gzip
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
import openai
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



# Model hyperparameters
sequence_length = 10

# Anomaly detection
def detect_anomalies(generator, discriminator, scaled_data_seq, features, threshold=0.5):
    features = features  
    # Generate fake sequences
    batch_size = scaled_data_seq.shape[0]
    sequence_length = 10
    random_latent_vectors = tf.random.normal(shape=(batch_size, sequence_length, features))
    generated_sequences = generator.predict(random_latent_vectors)
    
    # Get discriminator predictions for both real and fake data
    real_predictions = discriminator.predict(scaled_data_seq)
    fake_predictions = discriminator.predict(generated_sequences)
    
    # Identify real sequences that are classified as fake
    anomalies_indices = np.where(real_predictions.flatten() < threshold)[0]
    anomalies = scaled_data_seq[anomalies_indices]
    logging.info(f"Detected {len(anomalies)} potential anomalies.")
    
    return anomalies, real_predictions, fake_predictions


def query_model(messages):
    """
    Query the fine-tuned GPT-3.5 Turbo model with a prompt and return the response.
    """
    try:
        response = openai.ChatCompletion.create(
            model="ft:gpt-4o-2024-08-06:personal:pdm-genset-diaganostic-analytics:9z1ECxj4",
            messages=messages,
            temperature=1.0,
            max_tokens=150,
            stop=None
        )
        if response.choices and response.choices[0].message:
            return response.choices[0].message.content
        else:
            return "No response from the model."
            
    except Exception as e:
        print(f"Error querying model: {e}")
        
        return None


def generate_diagnosis_and_recommendation(anomaly_data):
    """
    Generate a prompt from anomaly data and query the model for diagnosis and recommendation.
    """
    if openai.api_key:
      conversation = [
          {"role": "system", "content": "You are an AI model specialized in anomaly detection and interpretation for generator data. Your task is to analyze provided anomaly data, interpret the underlying issues, and generate detailed, actionable recommendations for maintenance and improvement. Use clear, concise language and ensure your advice is practical and relevant to the context of the detected anomalies. Consider historical patterns, potential causes, and preventative measures in your recommendations."},
          {"role": "user", "content": f"Given the generator measurements: {anomaly_data}, what are the potential issues and recommended actions?"}
      ]
      try:
        # Query the model
        response = query_model(conversation)
      except openai.error.OpenAIError as e:
        st.error(f"Error: {e}")
    else:
      st.error("OpenAI API Key is missing. Please enter the API Key.")
        
    return response


def generate_prompts_from_anomalies(df):
    """
    Takes a DataFrame of anomalies and generates a list of prompts for each row.
    Each prompt includes the column values and a question about potential issues and recommended actions.
    
    Parameters:
    - anomalies_df: DataFrame containing anomaly data.
    
    Returns:
    - anomaly_data: A list of strings, each a prompt for a row in the DataFrame.
    """
    
    # Build the prompt string by iterating over each column and its value in the row
    anomaly_prompts = []
    
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Build the prompt string by iterating over each column and its value in the row
        prompt = ", ".join([f"{column} is {value}" for column, value in row.items()])
        # Append the question about potential issues and recommended actions
        prompt = prompt + ", what are the potential issues and recommended actions?"
        # Append the complete prompt to the list
        anomaly_prompts.append(prompt)
        
    return anomaly_prompts


def inverse_transform(scaled_data, scaler):
  return scaler.inverse_transform(scaled_data)


def create_sequences(data, seq_length):
    xs = []
    for i in range(len(data) - seq_length):
        sequence = data[i:(i + seq_length)]
        xs.append(sequence)
    return np.array(xs)


def load_model_from_github(url):
    filename = url.split('/')[-1]
    urllib.request.urlretrieve(url, filename)
    custom_objects = {'TransformerEncoderBlock': TransformerEncoderBlock}
    loaded_model = tf.keras.models.load_model(filename, custom_objects=custom_objects)
    return loaded_model


# Email alert function
def send_email(subject, body):
    sender_email = "pdm_genset_alerts_24@outlook.com"
    receiver_email = "tadiwanashe.nyaruwata@students.uz.ac.zw"
    password = "i^kbpbw7/3AVYA#"
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    message.attach(MIMEText(body, "plain"))

    with smtplib.SMTP("smtp-mail.outlook.com", 587) as server:
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())
