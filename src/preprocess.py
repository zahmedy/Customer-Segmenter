import pandas as pd
from sklearn.preprocessing import StandardScaler

# Features we decided to use for clustring
FEATURES = [
    "NumberPurchases",
    "AvgPurchaseValue",
    "SpendingScore",
    "Age",
    "Income"
]

def load_data(path: str) -> pd.DataFrame:
    """Load raw customer data from CSV."""
    df = pd.read_csv(path)
    return df

def preprocess(df: pd.DataFrame):
    """
    Select features and scale them with StandardScaler.
    Return:
        df          -> original dataframe (unchanged except maybe dropped rows)
        scaled_data -> numpy array with scaled features
        scaler      -> fitted StandardScaler
    """
    # (Optional) drop rows with missing values in the features we care about
    df = df.dropna(subset=FEATURES)

    scaler = StandardScaler()
    scaler.fit(df[FEATURES])
    scaled_data = scaler.transform(df[FEATURES])

    return df, scaled_data, scaler

if __name__ == "__main__":
    df = load_data("data/raw/customers.csv")
    df, scaled_data, scaler = preprocess(df)
    print("Scaled data shape:", scaled_data.shape)

