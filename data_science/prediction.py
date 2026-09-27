import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor


def predict_next_anomaly(data):

    if data is None or data.empty:
        raise ValueError("No market data available.")

    df = data.copy()

    required_columns = [
        "Daily_Return",
        "Return_Z",
        "Volume_Z",
        "Volume_Ratio",
        "Rolling_Volatility",
        "Anomaly_Score"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing prediction columns: "
            + ", ".join(missing)
        )

    features = [
        "Daily_Return",
        "Return_Z",
        "Volume_Z",
        "Volume_Ratio",
        "Rolling_Volatility",
        "Anomaly_Score"
    ]

    # Target = tomorrow's anomaly score
    df["Next_Anomaly"] = df["Anomaly_Score"].shift(-1)

    model_data = (
        df[features + ["Next_Anomaly"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(model_data) < 60:
        raise ValueError(
            "Not enough historical data for prediction."
        )

    X = model_data[features]
    y = model_data["Next_Anomaly"]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=4,
        random_state=42
    )

    model.fit(X, y)

    latest = (
        df[features]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .iloc[-1:]
    )

    if latest.empty:
        raise ValueError(
            "Latest market data is incomplete."
        )

    # Keep the exact same feature columns used during training
    latest_features = latest[X.columns]

    prediction = float(
        model.predict(latest_features)[0]
    )

    prediction = float(
        np.clip(prediction, 0, 100)
    )

    if prediction >= 75:
        classification = "Extreme"
    elif prediction >= 50:
        classification = "Unusual"
    else:
        classification = "Normal"

    # Estimate model confidence from tree disagreement
    tree_predictions = np.array([
    tree.predict(latest.values)[0]
    for tree in model.estimators_
])

    prediction_std = float(
        np.std(tree_predictions)
    )

    confidence = float(
        np.clip(
            100 - (prediction_std * 3),
            0,
            100
        )
    )

    return {
        "score": prediction,
        "classification": classification,
        "confidence": confidence
    }