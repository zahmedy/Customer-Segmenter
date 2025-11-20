# 🌀 Customer Segmenter – Unsupervised Learning with K-Means & PCA

Customer Segmenter is an end-to-end project that uses **unsupervised learning** to discover hidden customer groups based on their behavior and profile data.

It takes raw customer data → cleans and scales it → learns clusters with **K-Means** → reduces dimensionality with **PCA** → and visualizes the segments in an interactive **dashboard**.

---

## 🎯 Project Goals

- Learn and apply **unsupervised learning** (no labels).
- Group customers into meaningful **segments** based on:
  - Age  
  - Annual Income  
  - Spending Score  
  - Number of Purchases  
  - Average Purchase Value  
- Build a **visual dashboard** to:
  - Explore clusters in 2D (via PCA)
  - Inspect cluster-level statistics
  - Support basic business questions (e.g. “Who are my high-value customers?”)

---

## 🧠 Core Concepts

- **Unsupervised Learning**  
  No labels. The model discovers structure (clusters) directly from the feature space.

- **K-Means Clustering**
  - Uses **distance** between customers in feature space.
  - Groups similar customers into **k** clusters.
  - Optimizes **inertia** (sum of squared distances to cluster centers).

- **Feature Scaling (Standardization)**
  - Uses `StandardScaler` so all features have ~mean 0 and std 1.
  - Prevents large-range features (like income) from dominating distance.

- **PCA (Principal Component Analysis)**
  - Reduces high-dimensional data to **2 components** for visualization.
  - Preserves as much variance as possible in a lower-dimensional space.

---

## 🏗️ High-Level Architecture

1. **Data Layer**
   - `data/raw/` – raw customer CSV (e.g. `customers.csv`)
   - `data/processed/` – cleaned & scaled data, cluster assignments

2. **Preprocessing Pipeline**
   - Load data into a DataFrame.
   - Select numeric behavior features:
     - `Age`
     - `AnnualIncome`
     - `SpendingScore`
     - `NumPurchases`
     - `AvgPurchaseValue`
   - Handle missing values (drop or impute).
   - Apply `StandardScaler` to selected features.

3. **Modeling Layer (Unsupervised)**
   - **K Selection (Elbow Method)**  
     - Loop over `k` (e.g. 1 to 10).  
     - Fit K-Means for each `k` on scaled data.  
     - Store and plot inertia vs. `k`.  
     - Choose `k` at the “elbow” point.
   - **Final K-Means Model**
     - Train K-Means with chosen `k`.
     - Assign each customer a `cluster_id`.
     - Persist:
       - Trained scaler
       - Trained K-Means model
       - Customer data with cluster labels

4. **Dimensionality Reduction Layer**
   - Fit **PCA** on the scaled features.
   - Transform data into 2D:
     - `pc1`, `pc2` (first and second principal components).
   - Attach PCA coordinates to each customer along with `cluster_id`.

5. **Visualization & Dashboard Layer**
   - Framework (example): **Streamlit** or **Plotly Dash**
   - Views:
     - **Cluster Scatter Plot**  
       - PCA 2D scatter where:
         - x-axis = PC1  
         - y-axis = PC2  
         - color = `cluster_id`  
     - **Cluster Summary Panel**
       - Mean stats per cluster:
         - Avg age, income, spending score, purchases, etc.
     - **Customer Drill-down**
       - Filter customers by cluster.
       - Show a table of members in the selected segment.

---

## 📁 Suggested Project Structure

```bash
customer-segmenter/
├── data/
│   ├── raw/
│   │   └── customers.csv
│   └── processed/
│       ├── scaled_features.npy
│       ├── customer_segments.csv
│       └── pca_components.npy
├── models/
│   ├── scaler.pkl
│   └── kmeans.pkl
├── notebooks/
│   └── 01_eda_and_elbow_method.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── cluster.py
│   ├── reduce_dimensionality.py
│   └── dashboard.py
├── app.py          # main entry point for dashboard (e.g. Streamlit)
├── requirements.txt
└── README.md
