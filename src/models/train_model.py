import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.data.prepare_data import prepare_train_data
from src.data.split_data import split_data
from src.features.feature_engineering import create_features

def evaluate(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    mask = y_true != 0
    mape = (abs((y_true[mask] - y_pred[mask]) / y_true[mask])).mean() * 100
    return rmse, mae, mape



def clean_feature_names(df):
    # Replace any character that is NOT a-zA-Z0-9_ with _
    df.columns = df.columns.str.replace(r"[^0-9a-zA-Z_]", "_", regex=True)
    return df

def train():
    df = prepare_train_data()
    train, val, test = split_data(df)

    train = create_features(train)
    val = create_features(val)

    # Drop rows with NaNs from lag/rolling
    train = train.dropna()
    val = val.dropna()

    features = [
        "store_nbr",
        "family",
        "dayofweek",
        "month",
        "quarter",
        "year",
        "is_weekend",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_28",
        "onpromotion",
    ]

    X_train = pd.get_dummies(train[features], columns=["family"], drop_first=True)
    y_train = train["sales"]

    X_val = pd.get_dummies(val[features], columns=["family"], drop_first=True)
    y_val = val["sales"]

    # 🔥 CLEAN FEATURE NAMES (THIS FIXES YOUR ERROR)
    X_train = clean_feature_names(X_train)
    X_val = clean_feature_names(X_val)

    # Align columns
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.1,
        "num_leaves": 31,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        num_boost_round=500,
        callbacks=[
            lgb.early_stopping(20),
            lgb.log_evaluation(50)
        ],
    )

    y_pred = model.predict(X_val)
    rmse, mae, mape = evaluate(y_val, y_pred)

    print("RMSE:", rmse)
    print("MAE:", mae)
    print("MAPE:", mape)

    # Save model
    joblib.dump(model, "models/lightgbm_model.pkl")

if __name__ == "__main__":
    train()
