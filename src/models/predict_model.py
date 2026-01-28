# src/models/predict_model.py
import pandas as pd
import lightgbm as lgb
from src.data.prepare_data import load_raw_data
from src.features.feature_engineering import create_features

def predict():
    # --- Paso 1: Cargar datos ---
    data = load_raw_data()
    train = data['train']
    test = data['test']

    # --- Paso 2: Preparar features ---
    # Concatenar train y test para generar las mismas features
    full = pd.concat([train, test], sort=False, ignore_index=True)

    # Convertir sales a float, fill NaNs con 0
    if 'sales' in full.columns:
        full['sales'] = pd.to_numeric(full['sales'], errors='coerce').fillna(0)
    if 'onpromotion' in full.columns:
        full['onpromotion'] = pd.to_numeric(full['onpromotion'], errors='coerce').fillna(0)

    # Crear features con tu función
    full_features = create_features(full)

    # Separar de nuevo en train/test
    train_features = full_features[full_features['date'] <= train['date'].max()]
    test_features = full_features[full_features['date'] > train['date'].max()]

    # --- Paso 3: Definir columnas a usar para predicción ---
    features = [
        'store_nbr', 'family', 'onpromotion',
        'dayofweek', 'quarter', 'is_weekend',
        'lag_7', 'lag_14', 'lag_28',
        'rolling_mean_7', 'rolling_mean_28',
        'month', 'year'
    ]

    # Comprobar que las columnas existen
    features = [f for f in features if f in train_features.columns]

    # One-hot encoding para la columna 'family'
    X_test = pd.get_dummies(test_features[features], columns=['family'], drop_first=True)

    # --- Paso 4: Cargar modelo LightGBM ---
    import joblib
    model = joblib.load('models/lightgbm_model.pkl')


    # --- Paso 5: Predecir ---
    y_pred = model.predict(X_test)

    # --- Paso 6: Guardar predicciones ---
    submission = test_features[['id']].copy()
    submission['sales'] = y_pred
    submission.to_csv('predictions.csv', index=False)
    print("Predictions saved to predictions.csv")

if __name__ == "__main__":
    predict()
