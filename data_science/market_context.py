import yfinance as yf
import pandas as pd


# ==========================================
# FINESIGHT - MARKET CONTEXT ANALYSIS
# ==========================================

stock_ticker = "TCS.NS"
market_ticker = "^NSEI"   # NIFTY 50


# ==========================================
# 1. DOWNLOAD DATA
# ==========================================

stock = yf.download(
    stock_ticker,
    period="1y",
    auto_adjust=False
)

market = yf.download(
    market_ticker,
    period="1y",
    auto_adjust=False
)


# ==========================================
# 2. CLEAN YFINANCE DATA
# ==========================================

if isinstance(stock.columns, pd.MultiIndex):
    stock = stock.xs(
        stock_ticker,
        axis=1,
        level="Ticker"
    )

if isinstance(market.columns, pd.MultiIndex):
    market = market.xs(
        market_ticker,
        axis=1,
        level="Ticker"
    )

stock.columns.name = None
market.columns.name = None


# ==========================================
# 3. CALCULATE DAILY RETURNS
# ==========================================

stock["Return"] = stock["Close"].pct_change()

market["Return"] = market["Close"].pct_change()


# ==========================================
# 4. COMBINE STOCK + MARKET
# ==========================================

comparison = pd.DataFrame({
    "Stock_Return": stock["Return"],
    "Market_Return": market["Return"]
}).dropna()


# ==========================================
# 5. CALCULATE DIFFERENCE
# ==========================================

comparison["Relative_Performance"] = (
    comparison["Stock_Return"]
    - comparison["Market_Return"]
)


# ==========================================
# 6. CORRELATION
# ==========================================

correlation = comparison[
    "Stock_Return"
].corr(
    comparison["Market_Return"]
)


# ==========================================
# 7. BETA
# ==========================================

market_variance = comparison[
    "Market_Return"
].var()

beta = (
    comparison["Stock_Return"]
    .cov(comparison["Market_Return"])
    / market_variance
)


# ==========================================
# 8. LATEST MARKET MOVEMENT
# ==========================================

latest = comparison.iloc[-1]

stock_return = latest["Stock_Return"]
market_return = latest["Market_Return"]

relative_performance = (
    latest["Relative_Performance"]
)


# ==========================================
# 9. INTERPRETATION
# ==========================================

if (
    abs(stock_return) > 0.03
    and abs(stock_return) > abs(market_return) * 2
):
    context = "Potentially stock-specific movement"

elif (
    stock_return < 0
    and market_return < 0
):
    context = "Market-wide negative movement"

elif (
    stock_return > 0
    and market_return > 0
):
    context = "Market-wide positive movement"

else:
    context = "Mixed market movement"


# ==========================================
# 10. OUTPUT
# ==========================================

print("\n==========================================")
print("       FINESIGHT MARKET CONTEXT")
print("==========================================")

print("\nStock:", stock_ticker)
print("Benchmark: NIFTY 50")

print(
    "\nLatest Stock Return:",
    round(stock_return * 100, 2),
    "%"
)

print(
    "Latest NIFTY Return:",
    round(market_return * 100, 2),
    "%"
)

print(
    "Relative Performance:",
    round(relative_performance * 100, 2),
    "%"
)

print(
    "\nStock-Market Correlation:",
    round(correlation, 3)
)

print(
    "Beta:",
    round(beta, 3)
)

print(
    "\nMarket Context:",
    context
)


# ==========================================
# 11. TOP STOCK-SPECIFIC MOVEMENTS
# ==========================================

comparison["Absolute_Relative_Performance"] = (
    comparison["Relative_Performance"].abs()
)

top_events = comparison.nlargest(
    10,
    "Absolute_Relative_Performance"
)

print("\n==========================================")
print("     TOP STOCK-SPECIFIC MOVEMENTS")
print("==========================================")

print(
    top_events[
        [
            "Stock_Return",
            "Market_Return",
            "Relative_Performance"
        ]
    ]
)


print("\n==========================================")
print("             ANALYSIS COMPLETE")
print("==========================================")