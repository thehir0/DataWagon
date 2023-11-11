import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_val(df: pd.DataFrame, features: list, labels: list, val_size: float, seed: int):
    X = df[features]
    y = df[labels]

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=seed, stratify=y)

    return X_train, X_val, y_train, y_val
