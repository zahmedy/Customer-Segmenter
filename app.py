import streamlit as st
import plotly.express as px
import pandas as pd

def load_data():
    return pd.read_csv("data/processed/customer_segments.csv")

df = load_data()

st.title("🌀 Customer Segmenter Dashboard")

# Scatter plot (PCA)
fig = px.scatter(
    df,
    x="pc1",
    y="pc2",
    color="cluster_id",
    hover_data=["CustomerID", "Age", "Income", "NumberPurchases"]
)

st.plotly_chart(fig)

# Cluster filter
selected_cluster = st.selectbox("Select Cluster", sorted(df["cluster_id"].unique()))

filtered = df[df["cluster_id"] == selected_cluster]

st.subheader(f"Customers in Cluster {selected_cluster}")
st.write(filtered[ "Age", "Income", "NumberPurchases", "AvgPurchaseValue", "SpendingScore"])
