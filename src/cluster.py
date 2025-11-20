from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import joblib

from preprocess import load_data, preprocess

def compute_elbow(scaled_data, k_main: int = 1, k_max: int = 10):
    """
    Compute inertia values for different k to use in the elbow method.
    Returns:
        ks          -> list of k values
        inertias    -> list of inertia values for each k
    """
    ks = list(range(k_main, k_max + 1))
    inertias = []

    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        model.fit(scaled_data)
        inertias.append(model.inertia_)

    return ks, inertias

def plot_elbow(ks, inertias):
    """
    Plot inertia vs k to help choose the number of clusters 
    """
    plt.figure()
    plt.plot(ks, inertias, marker="o")
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of clusters K")
    plt.ylabel("Inertia (Within-cluster SSE)")
    plt.show()

def train_final_kmeans(scaled_data, n_clusters: int):
    """Train the final K-Means model on scaled data."""
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    model.fit(scaled_data)
    return model

if __name__ == "__main__":
    # 1. Load the customer data
    df = load_data("data/raw/customers.csv")
    df, scaled_data, scaler = preprocess(df)

    # 2. Compute elbow curve
    ks, inertias = compute_elbow(scaled_data)
    plot_elbow(ks, inertias)

    # Manually inspect the plot, then select the optimial K
    chosen_k = 10

    # 3. train final model
    kmeans = train_final_kmeans(scaled_data, chosen_k)

    # 4. Attach cluster labels to Dataframe
    df['cluster_id'] = kmeans.labels_

    # 5. Save outputs
    df.to_csv("data/processed/customer_segments.csv", index=False)
    joblib.dump(kmeans, "models/kmeans.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    
    print("Saved customer_segments.csv, kmeans.pkl, scaler.pkl")