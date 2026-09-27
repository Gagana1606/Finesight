import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. DOWNLOAD MARKET DATA
# ==========================================

ticker = "TCS.NS"

data = yf.download(
    ticker,
    period="1y",
    auto_adjust=False
)


# ==========================================
# 2. CLEAN YFINANCE MULTIINDEX
# ==========================================

if isinstance(data.columns, pd.MultiIndex):
    data = data.xs(ticker, axis=1, level="Ticker")

data.columns.name = None


# ==========================================
# 3. BASIC DATA CHECK
# ==========================================

print("\nDataset Shape:")
print(data.shape)

print("\nColumns:")
print(data.columns)

print("\nMissing Values:")
print(data.isnull().sum())


# ==========================================
# 4. DAILY RETURNS
# ==========================================

data["Daily_Return"] = data["Close"].pct_change()


# ==========================================
# 5. VOLATILITY
# ==========================================

daily_volatility = data["Daily_Return"].std()

annualized_volatility = (
    daily_volatility * np.sqrt(252)
)

print("\nDaily Volatility:", daily_volatility)

print(
    "Annualized Volatility:",
    annualized_volatility
)


# ==========================================
# 6. CUMULATIVE RETURN
# ==========================================

cumulative_return = (
    data["Adj Close"].iloc[-1]
    / data["Adj Close"].iloc[0]
) - 1

print(
    "Cumulative Return:",
    cumulative_return
)


# ==========================================
# 7. MAXIMUM DRAWDOWN
# ==========================================

rolling_max = data["Adj Close"].cummax()

drawdown = (
    data["Adj Close"] - rolling_max
) / rolling_max

max_drawdown = drawdown.min()

print(
    "Maximum Drawdown:",
    max_drawdown
)


# ==========================================
# 8. ANNUALIZED RETURN
# ==========================================

annualized_return = (
    data["Daily_Return"].mean() * 252
)

print(
    "Annualized Return:",
    annualized_return
)


# ==========================================
# 9. SHARPE RATIO
# ==========================================

sharpe_ratio = (
    annualized_return
    / annualized_volatility
)

print(
    "Sharpe Ratio:",
    sharpe_ratio
)


# ==========================================
# 10. PRICE TREND
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    data.index,
    data["Adj Close"]
)

plt.title(
    "TCS Adjusted Closing Price - 1 Year"
)

plt.xlabel("Date")
plt.ylabel("Adjusted Close Price")

plt.xticks(rotation=45)

plt.grid(True, alpha=0.25)

plt.tight_layout()

plt.show()


# ==========================================
# 11. DAILY RETURN DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 5))

plt.hist(
    data["Daily_Return"].dropna(),
    bins=30
)

plt.title(
    "TCS Daily Return Distribution"
)

plt.xlabel("Daily Return")
plt.ylabel("Frequency")

plt.grid(True, alpha=0.25)

plt.tight_layout()

plt.show()


# ==========================================
# 12. TRADING VOLUME
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    data.index,
    data["Volume"]
)

plt.title(
    "TCS Trading Volume"
)

plt.xlabel("Date")
plt.ylabel("Volume")

plt.xticks(rotation=45)

plt.grid(True, alpha=0.25)

plt.tight_layout()

plt.show()


# ==========================================
# 13. TOP 5 HIGH-VOLUME DAYS
# ==========================================

top_volume_days = data.nlargest(
    5,
    "Volume"
)[
    ["Volume", "Daily_Return"]
]

print("\nTop 5 Volume Days:")

print(top_volume_days)


# ==========================================
# 14. POSITIVE VS NEGATIVE
#     HIGH-VOLUME DAYS
# ==========================================

high_volume = data.nlargest(
    5,
    "Volume"
)

positive_days = (
    high_volume["Daily_Return"] > 0
).sum()

negative_days = (
    high_volume["Daily_Return"] < 0
).sum()

print(
    "\nPositive high-volume days:",
    positive_days
)

print(
    "Negative high-volume days:",
    negative_days
)


# ==========================================
# 15. VOLUME CHANGE
# ==========================================

data["Volume_Change"] = (
    data["Volume"].pct_change()
)


# ==========================================
# 16. VOLUME-RETURN CORRELATION
# ==========================================

correlation_data = data[
    ["Volume_Change", "Daily_Return"]
].dropna()

correlation = (
    correlation_data["Volume_Change"]
    .corr(correlation_data["Daily_Return"])
)

print(
    "\nVolume-Return Correlation:",
    correlation
)