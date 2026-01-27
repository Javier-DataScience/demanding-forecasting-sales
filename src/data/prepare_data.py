import pandas as pd
from src.data.load_data import load_raw_data

def prepare_train_data():
    data = load_raw_data()
    train = data["train"]

    # Agregar por date + store + family
    df = (
        train.groupby(["date", "store_nbr", "family"])
        .agg({"sales": "sum", "onpromotion": "sum"})
        .reset_index()
    )

    return df

if __name__ == "__main__":
    df = prepare_train_data()
    print(df.head())
    print(df.shape)
