import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def prepare_lstm_data(df, seq_len=10):
    features = ["supplier_score", "lead_time_days", "units_demanded"]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])

    X, y = [], []
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i - seq_len:i])
        y.append(scaled_data[i])

    return np.array(X), np.array(y), scaler

X, y, scaler = prepare_lstm_data(df)

# Build and train LSTM
model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
    Dense(X.shape[2])
])
model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=15, batch_size=16, verbose=0)

# Predict and calculate reconstruction error
y_pred = model.predict(X)
mse = np.mean(np.power(y - y_pred, 2), axis=1)
threshold = np.percentile(mse, 95)
anomaly_flags = mse > threshold

# Add back to df
df_lstm = df.copy()
df_lstm["lstm_anomaly"] = 0
df_lstm.loc[df_lstm.index[-len(anomaly_flags):], "lstm_anomaly"] = anomaly_flags.astype(int)
