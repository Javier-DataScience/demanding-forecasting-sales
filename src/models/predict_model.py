import joblib
import pandas as pd

from src.data.load_data import load_raw_data
from src.features.feature_engineering import create_features

def clean_feature_names(df):
    df.columns = df.columns.str.replace(r"[^0-9a-zA-Z_]", "_", regex=True)
    return df

def predict():
    # 1️⃣ Cargar modelo
    model = joblib.load("models/lightgbm_model.pkl")

    # 2️⃣ Cargar datos
    data = load_raw_data()
    train = data["train"]
    test = data["test"]
    test_ids = test["id"].copy()

    # 3️⃣ Concatenar train + test para poder calcular lags
    # Ponemos sales en test como NaN
    test["sales"] = pd.NA
    full = pd.concat([train, test], ignore_index=True)

    # 4️⃣ Crear features (lags y rolling)
    full_features = create_features(full)

    # 5️⃣ Tomar solo las filas del test
    test_features = full_features[full_features["sales"].isna()]

    # 6️⃣ Seleccionar columnas de features
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

    X_test = pd.get_dummies(test_features[features], columns=["family"], drop_first=True)
    X_test = clean_feature_names(X_test)

    # Alinear columnas con entrenamiento
    model_features = model.feature_name()
    X_test = X_test.reindex(columns=model_features, fill_value=0)

    # 7️⃣ Hacer predicciones
    y_pred = model.predict(X_test)

    # 8️⃣ Guardar resultados
    results = pd.DataFrame({
        "id": test_ids,
        "sales_pred": y_pred
    })

    results.to_csv("predictions.csv", index=False)
    print("Predicciones guardadas en predictions.csv")

if __name__ == "__main__":
    predict()
