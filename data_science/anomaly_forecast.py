import pandas as pd
import numpy as np


def create_forecasting_dataset(data):
    """
    Create a dataset where today's market information
    is used to predict whether tomorrow will be anomalous.
    """

    df = data.copy()

    # -------------------------------------------------
    # TARGET
    # -------------------------------------------------
    # Tomorrow's anomaly score
    df["Tomorrow_Anomaly_Score"] = (
        df["Anomaly_Score"].shift(-1)
    )

    # -------------------------------------------------
    # BINARY TARGET
    # -------------------------------------------------
    # Anomaly = score >= 50
    df["Tomorrow_Anomaly"] = (
        df["Tomorrow_Anomaly_Score"] >= 50
    ).astype(int)

    # -------------------------------------------------
    # FEATURES
    # -------------------------------------------------

    df["Return_5D"] = (
        df["Close"].pct_change(5)
    )

    df["Return_20D"] = (
        df["Close"].pct_change(20)
    )

    df["Price_vs_MA20"] = (
        df["Close"] / df["MA20"] - 1
    )

    df["Volume_Ratio"] = (
        df["Volume"]
        / df["Volume_Average_20"]
    )

    # -------------------------------------------------
    # RECENT VOLATILITY
    # -------------------------------------------------

    df["Volatility_5D"] = (
        df["Daily_Return"]
        .rolling(5)
        .std()
    )

    df["Volatility_20D"] = (
        df["Daily_Return"]
        .rolling(20)
        .std()
    )

    # -------------------------------------------------
    # RECENT ANOMALY ACTIVITY
    # -------------------------------------------------

    df["Previous_Anomaly"] = (
        df["Anomaly_Score"]
        .shift(1)
    )

    df["Anomalies_Last_5D"] = (
        df["Anomaly_Score"]
        .shift(1)
        .rolling(5)
        .apply(
            lambda x: np.sum(x >= 50),
            raw=True
        )
    )

    # -------------------------------------------------
    # FEATURES USED BY THE MODEL
    # -------------------------------------------------

    feature_columns = [
        "Daily_Return",
        "Return_5D",
        "Return_20D",
        "Volume_Ratio",
        "Return_Z",
        "Volume_Z",
        "Rolling_Volatility",
        "Price_vs_MA20",
        "Volatility_5D",
        "Volatility_20D",
        "Previous_Anomaly",
        "Anomalies_Last_5D"
    ]

    # -------------------------------------------------
    # REMOVE INVALID ROWS
    # -------------------------------------------------

    dataset = df[
        feature_columns
        + [
            "Tomorrow_Anomaly",
            "Tomorrow_Anomaly_Score"
        ]
    ].dropna()

    return dataset