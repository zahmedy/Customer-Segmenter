
import pandas as pd
from sklearn.decomposition import PCA
import joblib

from preprocess import load_data, preprocess
from cluster import train_final_kmeans
from preprocess import FEATURES  # same FEATURES list we used before

def apply_pca(scaled_data, n_components: int = 2):
    """
    Fit PCA on scaled data and return:
        components -> numpy array of shape (n_samples, n_components)
        pca        -> fitted PCA object
    """
    pca = PCA(n_components)
    components = pca.fit_transform(scaled_data)
    return components, pca


if __name__ == "__main__":
    # 1. Load the prepocess data
    df = load_data("data/processed/customer_segments.csv")
    df, scaled_data, scaler = preprocess(df)

    # 2. Load the same scaler we used earlier
    scaler = joblib.load("models/scaler.pkl")

    # 3. Scale features again using the saved scaler
    scaled_data = scaler.transform(df[FEATURES])

    # 4. Apply PCA
    components, pca_model = apply_pca(scaled_data, n_components=2)

    # 5. Attach PCA components
    df["pc1"] = components[:, 0]
    df["pc2"] = components[:, 1]

    # 6. Save updated dataframe and PCA model
    df.to_csv("data/processed/customer_segments_pca.csv", index=False)
    joblib.dump(pca_model, "models/pca.pkl")

    print("Saved customer_segments_pca.csv and pca.pkl")