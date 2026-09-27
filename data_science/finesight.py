import yfinance as yf
import numpy as np
import pandas as pd


# ==========================================
# FINESIGHT - MAIN ANALYSIS ENGINE
# ==========================================

ticker = "TCS.NS"
benchmark = "^NSEI"


# ==========================================
# 1. DOWNLOAD DATA
# ==========================================

stock = yf.download(
    ticker,
    period="1y",
    auto_adjust=False
)

market = yf.download(
    benchmark,
    period="1y",
    auto_adjust=False
)


# ==========================================
# 2. CLEAN YFINANCE DATA
# ==========================================

if isinstance(stock.columns, pd.MultiIndex):
    stock = stock.xs(
        ticker,
        axis=1,
        level="Ticker"
    )

if isinstance(market.columns, pd.MultiIndex):
    market = market.xs(
        benchmark,
        axis=1,
        level="Ticker"
    )

stock.columns.name = None
market.columns.name = None


# ==========================================
# 3. STOCK RETURNS
# ==========================================

stock["Daily_Return"] = (
    stock["Close"].pct_change()
)

market["Daily_Return"] = (
    market["Close"].pct_change()
)


# ==========================================
# 4. FINANCIAL FEATURES
# ==========================================

stock["Volume_Average_20"] = (
    stock["Volume"].rolling(20).mean()
)

stock["Volume_Ratio"] = (
    stock["Volume"]
    / stock["Volume_Average_20"]
)

stock["Rolling_Volatility"] = (
    stock["Daily_Return"].rolling(20).std()
)

return_mean = (
    stock["Daily_Return"].rolling(20).mean()
)

return_std = (
    stock["Daily_Return"].rolling(20).std()
)

stock["Return_ZScore"] = (
    (stock["Daily_Return"] - return_mean)
    / return_std
)

volume_mean = (
    stock["Volume"].rolling(20).mean()
)

volume_std = (
    stock["Volume"].rolling(20).std()
)

stock["Volume_ZScore"] = (
    (stock["Volume"] - volume_mean)
    / volume_std
)


# ==========================================
# 5. ANOMALY SCORE
# ==========================================

stock["Anomaly_Score"] = (
    0.6 * stock["Return_ZScore"].abs()
    +
    0.4 * stock["Volume_ZScore"].abs()
)

stock["Anomaly_Score"] = (
    stock["Anomaly_Score"]
    .clip(0, 4)
    / 4
    * 100
)


# ==========================================
# 6. ANOMALY CLASSIFICATION
# ==========================================

def classify_anomaly(score):

    if score >= 75:
        return "Extreme"

    elif score >= 50:
        return "Unusual"

    else:
        return "Normal"


stock["Anomaly_Level"] = (
    stock["Anomaly_Score"]
    .apply(classify_anomaly)
)


# ==========================================
# 7. COMPARE WITH NIFTY
# ==========================================

comparison = pd.DataFrame({
    "Stock_Return": stock["Daily_Return"],
    "Market_Return": market["Daily_Return"]
}).dropna()

comparison["Relative_Performance"] = (
    comparison["Stock_Return"]
    - comparison["Market_Return"]
)


# ==========================================
# 8. MERGE MARKET CONTEXT
# ==========================================

stock = stock.join(
    comparison["Market_Return"],
    how="left"
)

stock["Relative_Performance"] = (
    stock["Daily_Return"]
    - stock["Market_Return"]
)


# ==========================================
# 9. FIND MOST EXTREME EVENT
# ==========================================

valid_data = stock.dropna(
    subset=[
        "Anomaly_Score",
        "Relative_Performance"
    ]
)

event = valid_data.loc[
    valid_data["Anomaly_Score"].idxmax()
]


# ==========================================
# 10. EVENT VALUES
# ==========================================

event_date = event.name

event_return = event["Daily_Return"]

event_volume = event["Volume"]

event_volume_ratio = event["Volume_Ratio"]

event_return_z = event["Return_ZScore"]

event_volume_z = event["Volume_ZScore"]

event_score = event["Anomaly_Score"]

event_market_return = event["Market_Return"]

event_relative = event["Relative_Performance"]


# ==========================================
# 11. DETERMINE MARKET CONTEXT
# ==========================================

if (
    abs(event_return) >= 0.03
    and abs(event_relative) >= 0.03
):

    context = "Potentially stock-specific movement"

elif (
    event_return < 0
    and event_market_return < 0
):

    context = "Market-wide negative movement"

elif (
    event_return > 0
    and event_market_return > 0
):

    context = "Market-wide positive movement"

else:

    context = "Mixed market movement"


# ==========================================
# 12. DETERMINE REASONS
# ==========================================

reasons = []

if abs(event_return_z) >= 2:

    reasons.append(
        "Price movement was significantly larger than normal"
    )

if event_volume_z >= 2:

    reasons.append(
        "Trading volume was significantly higher than normal"
    )

if abs(event_relative) >= 0.03:

    reasons.append(
        "Stock movement was substantially different from NIFTY"
    )

if event_return < 0:

    reasons.append(
        "The movement was strongly negative"
    )

else:

    reasons.append(
        "The movement was strongly positive"
    )


# ==========================================
# 13. FINAL FINESIGHT REPORT
# ==========================================

print("\n")
print("==========================================")
print("             FINESIGHT ALERT")
print("==========================================")

print("\nStock:", ticker)

print(
    "Date:",
    event_date.date()
)

print(
    "\nPrice Movement:",
    round(event_return * 100, 2),
    "%"
)

print(
    "Volume:",
    int(event_volume)
)

print(
    "Volume vs Normal:",
    round(event_volume_ratio, 2),
    "x"
)

print(
    "Anomaly Score:",
    round(event_score, 2),
    "/ 100"
)

print(
    "Classification:",
    event["Anomaly_Level"]
)


print("\n------------------------------------------")
print("             MARKET CONTEXT")
print("------------------------------------------")

print(
    "\nTCS Return:",
    round(event_return * 100, 2),
    "%"
)

print(
    "NIFTY Return:",
    round(event_market_return * 100, 2),
    "%"
)

print(
    "Relative Performance:",
    round(event_relative * 100, 2),
    "%"
)

print(
    "\nAssessment:",
    context
)


print("\n------------------------------------------")
print("             WHY FLAGGED")
print("------------------------------------------")

for reason in reasons:

    print("•", reason)


print("\n==========================================")
print("          FINESIGHT ANALYSIS COMPLETE")
print("==========================================")