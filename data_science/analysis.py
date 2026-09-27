import pandas as pd
import numpy as np


def calculate_returns(data):
    """
    Calculate daily percentage returns.
    """

    data = data.copy()

    data["Daily_Return"] = (
        data["Close"].pct_change()
    )

    return data


def calculate_moving_averages(data):
    """
    Calculate short-term and medium-term
    moving averages.
    """

    data = data.copy()

    data["MA5"] = (
        data["Close"]
        .rolling(5)
        .mean()
    )

    data["MA20"] = (
        data["Close"]
        .rolling(20)
        .mean()
    )

    return data


def calculate_volume_analysis(data):
    """
    Calculate normal trading volume and
    volume ratio.
    """

    data = data.copy()

    data["Volume_Average_20"] = (
        data["Volume"]
        .rolling(20)
        .mean()
    )

    data["Volume_Ratio"] = (
        data["Volume"]
        /
        data["Volume_Average_20"]
    )

    return data


def calculate_volatility(data):
    """
    Calculate rolling 20-day volatility.
    """

    data = data.copy()

    data["Rolling_Volatility"] = (
        data["Daily_Return"]
        .rolling(20)
        .std()
    )

    return data


def calculate_return_zscore(data):
    """
    Measure how unusual the current return
    is compared with recent returns.
    """

    data = data.copy()

    rolling_mean = (
        data["Daily_Return"]
        .rolling(20)
        .mean()
    )

    rolling_std = (
        data["Daily_Return"]
        .rolling(20)
        .std()
    )

    data["Return_Z"] = (
        (
            data["Daily_Return"]
            - rolling_mean
        )
        /
        rolling_std.replace(
            0,
            np.nan
        )
    )

    return data


def calculate_volume_zscore(data):
    """
    Measure how unusual the current trading
    volume is compared with recent volume.
    """

    data = data.copy()

    rolling_mean = (
        data["Volume"]
        .rolling(20)
        .mean()
    )

    rolling_std = (
        data["Volume"]
        .rolling(20)
        .std()
    )

    data["Volume_Z"] = (
        (
            data["Volume"]
            - rolling_mean
        )
        /
        rolling_std.replace(
            0,
            np.nan
        )
    )

    return data


def calculate_anomaly_score(data):
    """
    Combine price-return and volume anomalies
    into a single score from 0 to 100.

    Return anomaly = 60%
    Volume anomaly = 40%
    """

    data = data.copy()

    data["Anomaly_Raw"] = (
        0.6
        * data["Return_Z"].abs()
        +
        0.4
        * data["Volume_Z"].abs()
    )

    data["Anomaly_Score"] = (
        data["Anomaly_Raw"]
        .clip(
            lower=0,
            upper=4
        )
        / 4
        * 100
    )

    return data


def classify_anomaly(score):
    """
    Convert numerical anomaly score
    into an understandable category.
    """

    if score >= 75:

        return "Extreme"

    elif score >= 50:

        return "Unusual"

    else:

        return "Normal"


def add_anomaly_classification(data):
    """
    Add anomaly classification to the dataset.
    """

    data = data.copy()

    data["Classification"] = (
        data["Anomaly_Score"]
        .apply(
            classify_anomaly
        )
    )

    return data


def run_analysis(data):
    """
    Run the complete Finesight
    quantitative analysis pipeline.
    """

    data = calculate_returns(
        data
    )

    data = calculate_moving_averages(
        data
    )

    data = calculate_volume_analysis(
        data
    )

    data = calculate_volatility(
        data
    )

    data = calculate_return_zscore(
        data
    )

    data = calculate_volume_zscore(
        data
    )

    data = calculate_anomaly_score(
        data
    )

    data = add_anomaly_classification(
        data
    )

    return data