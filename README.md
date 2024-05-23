# Predictive Maintenance System for IIoT Devices using Real-time Data Analytics and Anomaly Detection

## Project Description

This project is a Streamlit application designed to monitor and analyze real-time data from Industrial Internet of Things (IIoT) devices, specifically focusing on generator data. The application uses an anomaly detection model to identify anomalies in the data and a finetuned GPT-3.5 Turbo model to interpret these anomalies and generate actionable recommendations.

## Features

- Real-time data generation and monitoring
- Anomaly detection using a custom-trained model
- Interpretation of anomalies using GPT-3.5 Turbo
- Visual representation of generator data
- Insight generation and display
- Email alerts for detected anomalies

## Usage

1. **Start/Stop Generator**: Use the sidebar button to start or stop the generator data simulation.
2. **Enter OpenAI API Key**: Input your OpenAI API key in the provided text input field in the sidebar.
3. **Monitor Data**: View real-time data visualizations and insights on the main dashboard.
4. **Anomaly Detection**: The system will automatically detect anomalies and display relevant insights and recommendations.

## Configuration

### API Keys

Ensure you have set your OpenAI API key in the `.env` file:

```bash
OPENAI_API_KEY=your_openai_api_key
