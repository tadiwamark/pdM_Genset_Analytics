# generator_script.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_parameter_value(range_min, range_max, anomaly_factor=1.0):
    return np.random.uniform(range_min, range_max) * anomaly_factor

def simulate_anomalies(simulated_data):
    anomaly_type = np.random.choice(['electrical', 'temperature', 'pressure', 'none'], p=[0.2, 0.2, 0.2, 0.4])
    if anomaly_type == 'electrical':
        simulated_data['AverageCurrent(A)'] *= np.random.uniform(1.2, 1.5)
    elif anomaly_type == 'temperature':
        simulated_data['ExhaustTemp(°C)'] += np.random.uniform(20, 70)
    elif anomaly_type == 'pressure':
        simulated_data['inLetPressure(KPa)'] *= np.random.uniform(0.5, 0.8)
    return simulated_data, anomaly_type

def generate_continuous_data():
    while True:
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=1)
        time_range = pd.date_range(start=start_time, end=end_time, freq='1min')
        data_records = []

        for current_time in time_range:
            simulated_data = {
                'Time': current_time.strftime('%m/%d/%Y %H:%M'),
                'AverageCurrent(A)': generate_parameter_value(150, 650),
                'Phase1Current(A)': generate_parameter_value(150, 650),
                'Phase2Current(A)': generate_parameter_value(150, 650),
                'Phase3Current(A)': generate_parameter_value(150, 650),
                'ExhaustTemp(°C)': generate_parameter_value(500, 750),
                'inLetPressure(KPa)': generate_parameter_value(25, 100),
                'outLetPressure(KPa)': generate_parameter_value(20, 90),
                'OutLetAirTemp(°C)': generate_parameter_value(20, 41),
                'CoolantTemp( °C)': generate_parameter_value(30, 95),
                'OilPressure(KPa)': generate_parameter_value(200, 500),
                'PowerFactor': generate_parameter_value(0.8, 1.2),
                'Speed(Rpm)': generate_parameter_value(1200, 1700),
                'AmbientTemp( °C)': generate_parameter_value(25, 35),
                'FuelLevel(Ltrs)': generate_parameter_value(1340,1500),
                'Freq(Hz)': generate_parameter_value(0, 60),
            }
            simulated_data, _ = simulate_anomalies(simulated_data)
            data_records.append(simulated_data)

        yield pd.DataFrame(data_records)
