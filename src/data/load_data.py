import pandas as pd
from pathlib import Path

def load_raw_data(data_dir: str = "data/raw"):
    """
    Load all raw CSV files into a dictionary of DataFrames.
    """
    data_path = Path(data_dir)

    train = pd.read_csv(data_path / "train.csv", parse_dates=["date"])
    test = pd.read_csv(data_path / "test.csv", parse_dates=["date"])
    stores = pd.read_csv(data_path / "stores.csv")
    transactions = pd.read_csv(data_path / "transactions.csv", parse_dates=["date"])
    oil = pd.read_csv(data_path / "oil.csv", parse_dates=["date"])
    
    # CORRECCIÓN: el archivo se llama "holidays_events.csv"
    holidays = pd.read_csv(data_path / "holidays_events.csv", parse_dates=["date"])
    
    sample_submission = pd.read_csv(data_path / "sample_submission.csv")

    return {
        "train": train,
        "test": test,
        "stores": stores,
        "transactions": transactions,
        "oil": oil,
        "holidays": holidays,
        "sample_submission": sample_submission,
    }

if __name__ == "__main__":
    data = load_raw_data()
    print("Loaded datasets:")
    for k, v in data.items():
        print(k, v.shape)
