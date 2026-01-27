from src.data.load_data import load_raw_data

def inspect():
    data = load_raw_data()
    train = data["train"]

    print("TRAIN COLUMNS:", train.columns)
    print("TRAIN INFO:")
    print(train.info())

    print("\nNULLS:")
    print(train.isna().sum())

    print("\nDATE RANGE:")
    print(train["date"].min(), train["date"].max())

    print("\nUNIQUE STORES:", train["store_nbr"].nunique())
    print("UNIQUE FAMILIES:", train["family"].nunique())

if __name__ == "__main__":
    inspect()
