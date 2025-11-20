# app.py
import streamlit as st
import plotly.express as px
import pandas as pd

def load_clustered_pca_data():
    """Load processed data with cluster_id, pc1, pc2."""
    return pd.read_csv("data/processed/customer_segments_pca.csv")

df = load_clustered_pca_data()

st.title("🌀 Customer Segmenter Dashboard")

# PCA Scatter Plot
fig = px.scatter(
    df,
    x="pc1",
    y="pc2",
    color="cluster_id",
    hover_data=["Age", "Income", "NumberPurchases", "AvgPurchaseValue"]
)

st.subheader("PCA Cluster Visualization")
st.plotly_chart(fig)

# Cluster Filter
clusters = sorted(df["cluster_id"].unique())
selected_cluster = st.selectbox("Select a cluster to inspect:", clusters)

cluster_df = df[df["cluster_id"] == selected_cluster]

st.subheader(f"Cluster {selected_cluster} – Customers")
st.dataframe(cluster_df[[
    "Age",
    "Income",
    "NumberPurchases",
    "AvgPurchaseValue",
    "SpendingScore"
]])
