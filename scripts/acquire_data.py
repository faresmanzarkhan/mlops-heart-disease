import os
import urllib.request
import pandas as pd

RAW_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COL_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

def acquire_data():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    raw_path = "data/raw/heart.data"
    
    print("Downloading UCI Heart Disease Cleveland dataset...")
    urllib.request.urlretrieve(RAW_DATA_URL, raw_path)
    
    # Read raw data (handling '?' missing values)
    df = pd.read_csv(raw_path, header=None, names=COL_NAMES, na_values="?")
    print(f"Raw data shape: {df.shape}")
    
    # Clean and map target to binary (0 = no disease, 1 = disease present 1-4)
    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)
    
    # Drop rows with missing values for clean training
    df_cleaned = df.dropna()
    print(f"Cleaned data shape: {df_cleaned.shape}")
    
    processed_path = "data/processed/heart_cleaned.csv"
    df_cleaned.to_csv(processed_path, index=False)
    print(f"Processed data saved to {processed_path}")

if __name__ == "__main__":
    acquire_data()
