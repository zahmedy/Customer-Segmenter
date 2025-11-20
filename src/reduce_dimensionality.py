
import pandas as pd
from sklearn.decomposition import PCA
import joblib

from preprocess import load_data, preprocess
from cluster import train_final_kmeans

def apply_pca(scaled_data, n_components: int = 2):
    """
    Fit PCA on scaled data and return:
        components -> numpy array of shape (n_samples, n_components)
        pca        -> fitted PCA object
    """
    pca = PCA(n_components)
    components = pca.fit_transform(scaled_data)
    return components, pca

