import pandas as pd

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df['sales'] = pd.to_numeric(df.get('sales', 0), errors='coerce').fillna(0)

    # Lags
    for lag in [7, 14, 28]:
        df[f'lag_{lag}'] = (
            pd.to_numeric(
                df.groupby(['store_nbr', 'family'])['sales']
                  .shift(lag),
                errors='coerce'
            ).fillna(0)
        )

    # Rolling means
    df['rolling_mean_7'] = (
        pd.to_numeric(
            df.groupby(['store_nbr', 'family'])['sales']
              .shift(1)
              .rolling(window=7, min_periods=1)
              .mean(),
            errors='coerce'
        ).fillna(0)
    )
    df['rolling_mean_28'] = (
        pd.to_numeric(
            df.groupby(['store_nbr', 'family'])['sales']
              .shift(1)
              .rolling(window=28, min_periods=1)
              .mean(),
            errors='coerce'
        ).fillna(0)
    )

    # Temporal features
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)

    # Fill remaining NA
    df.fillna(0, inplace=True)
    return df
