from src.data.prepare_data import prepare_train_data

def split_data(df):
    train = df[df["date"] < "2017-01-01"]
    val = df[(df["date"] >= "2017-01-01") & (df["date"] < "2017-06-01")]
    test = df[df["date"] >= "2017-06-01"]

    return train, val, test

if __name__ == "__main__":
    df = prepare_train_data()
    train, val, test = split_data(df)
    print("Train:", train.shape)
    print("Val:", val.shape)
    print("Test:", test.shape)
