#generator_script.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time


def generate_parameter_value(range_min, range_max, anomaly_factor=1.0):
    """
    Generates a random parameter value, optionally adjusted by an anomaly factor.
    """
    return np.random.uniform(range_min, range_max) * anomaly_factor


def simulate_anomalies(simulated_data):
    """
    Introduces anomalies into the simulated data based on predefined conditions.
    """
    anomaly_type = np.random.choice(['electrical', 'temperature', 'pressure', 'none'],
                                    p=[0.0, 0.0, 0.0, 1.0])  
    if anomaly_type == 'electrical':
        simulated_data['AverageCurrent(A)'] *= np.random.uniform(1.2, 1.5)
        simulated_data['Phase1Current(A)'] *= np.random.uniform(1.2, 1.5)
        simulated_data['Phase2Current(A)'] *= np.random.uniform(1.2, 1.5)
        simulated_data['Phase3Current(A)'] *= np.random.uniform(1.2, 1.5)
    elif anomaly_type == 'temperature':
        simulated_data['ExhaustTemp(°C)'] += np.random.uniform(20, 70)
    elif anomaly_type == 'pressure':
        simulated_data['inLetPressure(KPa)'] *= np.random.uniform(0.5, 0.8)
    return simulated_data, anomaly_type


def generate_continuous_data(start_time):
    print(f"Generating data from {start_time} with interval 5 seconds")

    while True:
        current_time = start_time + timedelta(seconds=5)
        data_records = []
        simulated_data = {
            'Time': current_time,
            'AverageCurrent(A)': generate_parameter_value(290, 400),
            'Phase1Current(A)': generate_parameter_value(350, 400),
            'Phase2Current(A)': generate_parameter_value(400, 420),
            'Phase3Current(A)': generate_parameter_value(380, 400),
            'ExhaustTemp(°C)': generate_parameter_value(480, 500),
            'inLetPressure(KPa)': generate_parameter_value(50, 70),
            'outLetPressure(KPa)': generate_parameter_value(50, 70),
            'OutLetAirTemp(°C)': generate_parameter_value(27, 29),
            'CoolantTemp( °C)': generate_parameter_value(60, 73),
            'OilPressure(KPa)': generate_parameter_value(340, 360),
            'PowerFactor': generate_parameter_value(1, 1),
            'Speed(Rpm)': generate_parameter_value(1450, 1470),
            'AmbientTemp(°C)': generate_parameter_value(29, 31),
            'FuelLevel(Ltrs)': generate_parameter_value(1200, 1250),
            'Freq(Hz)': generate_parameter_value(47, 50)
        }
        
        simulated_data, anomaly_type = simulate_anomalies(simulated_data)
        simulated_data['Anomaly_Type'] = anomaly_type
        data_records.append(simulated_data)
        simulated_df = pd.DataFrame(data_records)

        print(f"Generated {len(data_records)} records.")

        yield simulated_df

        start_time = current_time
        time.sleep(1)
