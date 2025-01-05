import streamlit as st
import pandas as pd
import numpy as np
import time
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the AML model
@st.cache_resource
def load_aml_model():
    model_path = "https://github.com/tadiwamark/AML/releases/download/dnn_aml/dnn_aml_model.h5"  # Replace with your model URL
    model = load_model(model_path)
    return model

# Preprocess data
def preprocess_data(data, scaler):
    # Feature engineering (example)
    data['Year'] = pd.to_datetime(data['Timestamp']).dt.year
    data['Month'] = pd.to_datetime(data['Timestamp']).dt.month
    data['Day'] = pd.to_datetime(data['Timestamp']).dt.day
    data['Hour'] = pd.to_datetime(data['Timestamp']).dt.hour
    data['Minute'] = pd.to_datetime(data['Timestamp']).dt.minute
    data = data.drop(columns=['Timestamp'])

    # Encoding and scaling
    categorical_columns = ['Receiving Currency', 'Payment Currency', 'Payment Format']
    for col in categorical_columns:
        data[col] = data[col].map({'USD': 0, 'EUR': 1, 'GBP': 2, 'Wire': 0, 'Credit Card': 1, 'Cheque': 2, 'Reinvestment': 3})

    data['Account'] = data['Account'].astype(str).apply(lambda x: hash(x) % (10**6))
    data['Account.1'] = data['Account.1'].astype(str).apply(lambda x: hash(x) % (10**6))

    numeric_columns = ['Amount Received', 'Amount Paid']
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    return data

# Generate transactions continuously
def generate_continuous_transactions():
    while True:
        batch = pd.DataFrame({
            'Timestamp': pd.date_range(start=pd.Timestamp.now(), periods=10, freq='S'),
            'From Bank': np.random.randint(1, 1000, 10),
            'Account': [f'8000{np.random.randint(1000, 9999)}' for _ in range(10)],
            'To Bank': np.random.randint(1, 1000, 10),
            'Account.1': [f'8000{np.random.randint(1000, 9999)}' for _ in range(10)],
            'Amount Received': np.random.uniform(0.01, 10000, 10),
            'Receiving Currency': np.random.choice(['USD', 'EUR', 'GBP'], 10),
            'Amount Paid': np.random.uniform(0.01, 10000, 10),
            'Payment Currency': np.random.choice(['USD', 'EUR', 'GBP'], 10),
            'Payment Format': np.random.choice(['Wire', 'Credit Card', 'Cheque', 'Reinvestment'], 10),
        })
        yield batch

# Streamlit app
def main():
    st.set_page_config(page_title="AML Monitoring", layout="wide")
    st.title("💸 Real-Time AML Monitoring System")

    # Load model and initialize scaler
    model = load_aml_model()
    scaler = StandardScaler()

    # Placeholders
    data_placeholder = st.empty()
    flagged_placeholder = st.empty()
    stats_placeholder = st.empty()
    graph_placeholder = st.empty()

    if "transactions" not in st.session_state:
        st.session_state["transactions"] = pd.DataFrame()
    if "flagged_transactions" not in st.session_state:
        st.session_state["flagged_transactions"] = pd.DataFrame()

    transaction_generator = generate_continuous_transactions()

    # Real-time monitoring loop
    while True:
        try:
            # Generate and update transactions
            new_batch = next(transaction_generator)
            st.session_state["transactions"] = pd.concat([st.session_state["transactions"], new_batch]).tail(100)

            # Display transactions
            data_placeholder.dataframe(st.session_state["transactions"])

            # Process every 60 records
            if len(st.session_state["transactions"]) >= 60:
                to_process = st.session_state["transactions"].tail(60)
                processed_data = preprocess_data(to_process.copy(), scaler)
                predictions = (model.predict(processed_data) > 0.5).astype(int)
                to_process['Is Laundering'] = predictions

                # Update flagged transactions
                flagged = to_process[to_process['Is Laundering'] == 1]
                st.session_state["flagged_transactions"] = pd.concat([st.session_state["flagged_transactions"], flagged])

                # Display flagged transactions
                flagged_placeholder.subheader("🚩 Flagged Suspicious Transactions")
                flagged_placeholder.dataframe(st.session_state["flagged_transactions"])

                # Display stats
                stats_placeholder.subheader("📊 Statistics")
                stats_placeholder.metric("Total Transactions", len(st.session_state["transactions"]))
                stats_placeholder.metric("Flagged Transactions", len(st.session_state["flagged_transactions"]))

                # Visualize flagged transactions
                if not st.session_state["flagged_transactions"].empty:
                    fig, ax = plt.subplots()
                    st.session_state["flagged_transactions"]['Amount Received'].hist(ax=ax, bins=20)
                    ax.set_title("Distribution of Amount Received (Flagged Transactions)")
                    ax.set_xlabel("Amount Received")
                    ax.set_ylabel("Frequency")
                    graph_placeholder.pyplot(fig)

            time.sleep(1)

        except StopIteration:
            break

if __name__ == "__main__":
    main()
