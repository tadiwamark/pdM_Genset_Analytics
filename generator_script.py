import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

class AnomalySimulator:
    def __init__(self):
        self.anomaly_active = False
        self.anomaly_type = 'none'
        self.anomaly_start = None
        self.anomaly_duration = 0
        self.anomaly_progress = 0

    def start_anomaly(self):
        self.anomaly_type = np.random.choice(['electrical', 'temperature', 'pressure'],
                                             p=[0.4, 0.3, 0.3])
        self.anomaly_duration = np.random.randint(10, 30)  # Anomaly duration in iterations
        self.anomaly_progress = 0
        self.anomaly_active = True

    def simulate_anomalies(self, simulated_data):
        if not self.anomaly_active and np.random.rand() < 0.1:  # 10% chance to start an anomaly
            self.start_anomaly()

        if self.anomaly_active:
            progress_ratio = self.anomaly_progress / self.anomaly_duration
            if self.anomaly_type == 'electrical':
                factor = 1 + progress_ratio * np.random.uniform(0.2, 0.5)
                simulated_data['AverageCurrent(A)'] *= factor
                simulated_data['Phase1Current(A)'] *= factor
                simulated_data['Phase2Current(A)'] *= factor
                simulated_data['Phase3Current(A)'] *= factor
            elif self.anomaly_type == 'temperature':
                increase = progress_ratio * np.random.uniform(20, 70)
                simulated_data['ExhaustTemp(°C)'] += increase
                simulated_data['CoolantTemp(°C)'] += increase * 0.5  # Related parameter
            elif self.anomaly_type == 'pressure':
                factor = 1 - progress_ratio * np.random.uniform(0.2, 0.5)
                simulated_data['inLetPressure(KPa)'] *= factor
                simulated_data['outLetPressure(KPa)'] *= factor

            self.anomaly_progress += 1
            if self.anomaly_progress >= self.anomaly_duration:
                self.anomaly_active = False
                self.anomaly_type = 'none'

        return simulated_data, self.anomaly_type


def generate_parameter_value(range_min, range_max, anomaly_factor=1.0):
    """
    Generates a random parameter value, optionally adjusted by an anomaly factor.
    """
    return np.random.uniform(range_min, range_max) * anomaly_factor


def generate_continuous_data(start_time):
    print(f"Generating data from {start_time} with interval 5 seconds")
    anomaly_simulator = AnomalySimulator()

    while True:
        current_time = start_time + timedelta(seconds=5)
        data_records = []
        simulated_data = {
            'Time': current_time,
            'AverageCurrent(A)': generate_parameter_value(380, 420),
            'Phase1Current(A)': generate_parameter_value(380, 420),
            'Phase2Current(A)': generate_parameter_value(380, 420),
            'Phase3Current(A)': generate_parameter_value(380, 420),
            'ExhaustTemp(°C)': generate_parameter_value(450, 500),
            'inLetPressure(KPa)': generate_parameter_value(45, 75),
            'outLetPressure(KPa)': generate_parameter_value(45, 75),
            'OutLetAirTemp(°C)': generate_parameter_value(25, 30),
            'CoolantTemp( °C)': generate_parameter_value(70, 85),
            'OilPressure(KPa)': generate_parameter_value(300, 360),
            'PowerFactor': generate_parameter_value(0.95, 1.0),
            'Speed(Rpm)': generate_parameter_value(1450, 1550),
            'AmbientTemp(°C)': generate_parameter_value(25, 35),
            'FuelLevel(Ltrs)': generate_parameter_value(1100, 1250),
            'Freq(Hz)': generate_parameter_value(48, 52)
        }

        simulated_data, anomaly_type = anomaly_simulator.simulate_anomalies(simulated_data)
        simulated_data['Anomaly_Type'] = anomaly_type
        data_records.append(simulated_data)
        simulated_df = pd.DataFrame(data_records)

        print(f"Generated {len(data_records)} records. Current anomaly: {anomaly_type}")

        yield simulated_df

        start_time = current_time
        time.sleep(5)  # Keep interval at 5 seconds
