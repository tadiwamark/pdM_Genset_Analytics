# -*- coding: utf-8 -*-
"""model_utils.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/10T8M-L93ZjoADnbICidHP1Fi4tN8MYGd
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import time
import urllib.request
from transformer_encoder_block import TransformerEncoderBlock
import gzip
import requests



# Anomaly detection
def detect_anomalies(generator, discriminator, data, threshold=0.5):


    # Scale the data

    scaled_data = scaler.transform(data[numerical_features])

    # Check if scaled_data contains any NaN values
    assert not np.isnan(scaled_data).any(), "Scaled data contains NaN after scaling"
    assert not np.isinf(scaled_data).any(), "Scaled data contains Inf after scaling"

    sequence_length = 10

    def create_sequences(data, seq_length):
        xs = []
        for i in range(len(data) - seq_length):
            sequence = data[i:(i + seq_length)]
            xs.append(sequence)
        return np.array(xs)

    scaled_data_seq = create_sequences(scaled_data, sequence_length)


    features = scaled_data.shape[1]


    # Generate fake sequences
    batch_size = data.shape[0]
    random_latent_vectors = tf.random.normal(shape=(batch_size, sequence_length, features))
    generated_sequences = generator.predict(random_latent_vectors)

    # Get discriminator predictions for both real and fake data
    real_predictions = discriminator.predict(data)
    fake_predictions = discriminator.predict(generated_sequences)

    # Identify real sequences that are classified as fake
    anomalies = data[real_predictions.flatten() < threshold]

    return anomalies, real_predictions, fake_predictions


def query_model(messages):
    """
    Query the fine-tuned GPT-3.5 Turbo model with a prompt and return the response.
    """

    try:
        response = client.chat.completions.create(
            model="ft:gpt-3.5-turbo-1106:personal::8rMBgWJN",
            messages=messages,
            temperature=1,
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
          {"role": "system", "content": "You are Jenny a generator expert who interprets generator data anomalies and gives us her expert recommendations."},
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
  anomaly_data = ", ".join([f"{column} is {value}" for column, value in row.items()])


  # Append the question about potential issues and recommended actions
  anomaly_data += " what are the potential issues and recommended actions?"




  return anomaly_data

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

def download_and_load_scaler(url):
    """
    Downloads a compressed (.gz) scaler from the given URL, decompresses it, and loads it using pickle.
    
    Parameters:
        url (str): The URL to download the compressed scaler from.
        
    Returns:
        scaler (sklearn.preprocessing.StandardScaler): The loaded scaler.
    """
    try:
        # Send a GET request to download the file
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful

        # Decompress and load the scaler
        with gzip.GzipFile(fileobj=response.raw) as gz:
            scaler = pickle.load(gz)
            return scaler
    except requests.RequestException as e:
        st.error(f'Failed to download the scaler file: {e}')
    except Exception as e:
        st.error(f'An error occurred while loading the scaler: {e}')

    return None
