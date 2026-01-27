# Demanding Forecasting Sales

## Objective
Build a reproducible local pipeline for forecasting sales using LightGBM.

## Dataset
- Kaggle: Demanding Forecasting Sales
- Columns: date, store, item, sales

## Pipeline
- Data ingestion
- Data cleaning
- Feature engineering
- Model training (LightGBM)
- Evaluation (RMSE, MAE, MAPE)
- Inference (forecast for next 28 days)

## Run
```bash
python src/models/train_model.py
