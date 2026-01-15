import pandas as pd


def split_series_last_n_weeks(ts: pd.Series, n_test: int, n_dev: int) -> dict:
    if n_test <= 0 or n_dev <= 0:
        raise ValueError("n_test and n_dev must be > 0")
    if n_test + n_dev >= len(ts):
        raise ValueError("Not enough data to split into train/dev/test with these sizes.")

    test = ts.iloc[-n_test:]
    dev = ts.iloc[-(n_test + n_dev) : -n_test]
    train = ts.iloc[: -(n_test + n_dev)]

    print("\n=== Time series splits ===")
    print(f"Train: {train.index[0].date()} → {train.index[-1].date()} ({len(train)} weeks)")
    print(f"Dev:   {dev.index[0].date()} → {dev.index[-1].date()} ({len(dev)} weeks)")
    print(f"Test:  {test.index[0].date()} → {test.index[-1].date()} ({len(test)} weeks)")

    return {"train": train, "dev": dev, "test": test}