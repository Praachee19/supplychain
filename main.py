import streamlit as st
import pandas as pd
import numpy as np
from faker import Faker
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

st.set_page_config(page_title="Supply Chain Disruption Detection", layout="wide")
st.title("🚨 Supply Chain Disruption Detection Dashboard")

# -----------------------------
# Step 1: Simulate synthetic data
# -----------------------------
@st.cache_data
def generate_data(n_days=90):
    fake = Faker()
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days)

    data = pd.DataFrame({
        "date": dates,
        "supplier_score": np.clip(np.random.normal(85, 5, n_days), 50, 100),
        "lead_time_days": np.clip(np.random.normal(7, 2, n_days), 2, 20),
        "units_demanded": np.random.poisson(lam=100, size=n_days),
        "country": [fake.country() for _ in range(n_days)],
    })

    # Inject disruption anomalies
    data.loc[np.random.choice(n_days, 5), "lead_time_days"] += np.random.randint(5, 15, 5)
    data.loc[np.random.choice(n_days, 4), "supplier_score"] -= np.random.randint(10, 30, 4)
    data.loc[np.random.choice(n_days, 4), "units_demanded"] += np.random.randint(100, 300, 4)

    return data

df = generate_data()

# -----------------------------
# Step 2: Detect anomalies
# -----------------------------
def detect_anomalies(df):
    df_model = df[["supplier_score", "lead_time_days", "units_demanded"]]
    model = IsolationForest(contamination=0.08, random_state=42)
    df["anomaly"] = model.fit_predict(df_model)
    df["anomaly"] = df["anomaly"].map({1: 0, -1: 1})  # 1 = anomaly
    return df

df = detect_anomalies(df)

# -----------------------------
# Step 3: Visualize
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Lead Time Trend")
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["lead_time_days"], label="Lead Time (days)")
    ax.scatter(df[df["anomaly"] == 1]["date"], df[df["anomaly"] == 1]["lead_time_days"], color='red', label="Disruption")
    ax.set_xlabel("Date")
    ax.set_ylabel("Lead Time")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Supplier Score Trend")
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["supplier_score"], label="Supplier Score", color='green')
    ax.scatter(df[df["anomaly"] == 1]["date"], df[df["anomaly"] == 1]["supplier_score"], color='red', label="Disruption")
    ax.set_xlabel("Date")
    ax.set_ylabel("Score")
    ax.legend()
    st.pyplot(fig)

# -----------------------------
# Step 4: Alerts
# -----------------------------
st.subheader("⚠️ Disruption Alerts")
alerts = df[df["anomaly"] == 1][["date", "supplier_score", "lead_time_days", "units_demanded", "country"]]
alerts.columns = ["Date", "Supplier Score", "Lead Time (days)", "Demand", "Country"]
st.dataframe(alerts.reset_index(drop=True))

# -----------------------------
# Step 5: Export (Optional)
# -----------------------------
st.download_button("📥 Download Alert Report", alerts.to_csv(index=False), file_name="supply_chain_alerts.csv")
