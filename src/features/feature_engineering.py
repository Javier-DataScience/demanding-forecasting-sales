import pandas as pd
from src.data.split_data import split_data
from src.data.prepare_data import prepare_train_data

def create_features(df):
    df = df.copy()
    
    # Temporal features
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # Lag features
    df["lag_7"] = df.groupby(["store_nbr", "family"])["sales"].shift(7)
    df["lag_14"] = df.groupby(["store_nbr", "family"])["sales"].shift(14)
    df["lag_28"] = df.groupby(["store_nbr", "family"])["sales"].shift(28)

    # Rolling features
    df["rolling_mean_7"] = (
        df.groupby(["store_nbr", "family"])["sales"]
        .shift(1)
        .rolling(window=7)
        .mean()
    )
    df["rolling_mean_28"] = (
        df.groupby(["store_nbr", "family"])["sales"]
        .shift(1)
        .rolling(window=28)
        .mean()
    )

    return df

if __name__ == "__main__":
    df = prepare_train_data()
    train, val, test = split_data(df)

    train = create_features(train)
    val = create_features(val)
    test = create_features(test)

    print("Train features:", train.shape)
    print("Val features:", val.shape)
    print("Test features:", test.shape)
