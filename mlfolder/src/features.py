import pandas as pd


def make_ml_features(ts: pd.Series, max_lag: int = 12) -> pd.DataFrame:
    df = pd.DataFrame({"y": ts})

    for lag in range(1, max_lag + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)

    df["ma_4"] = df["y"].rolling(4).mean()
    df["ma_8"] = df["y"].rolling(8).mean()
    df["ma_12"] = df["y"].rolling(12).mean()

    df["weekofyear"] = df.index.isocalendar().week.astype(int)
    df["month"] = df.index.month
    df["year"] = df.index.year

    return df.dropna()
