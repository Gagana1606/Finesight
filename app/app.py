import sys
import re
from pathlib import Path

# =========================================================
# PROJECT ROOT
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# =========================================================
# IMPORTS
# =========================================================

import requests
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

from data_science.prediction import predict_next_anomaly
from data_science.news import get_news
from data_science.nlp import analyze_news


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Finesight",
    page_icon="📈",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("📈 Finesight")

st.caption(
    "Intelligent Stock Movement & Anomaly Analysis"
)

st.write(
    "Detect → Measure → Compare → Explain"
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("🔎 Stock Analysis")

user_input = st.sidebar.text_input(
    "Enter stock ticker or company name",
    value="TCS",
    placeholder="Example: TCS, Apple, Microsoft, Tata Steel"
)

period = st.sidebar.selectbox(
    "Analysis Period",
    [
        "6mo",
        "1y",
        "2y",
        "5y"
    ],
    index=1
)


# =========================================================
# NORMALIZE SEARCH INPUT
# =========================================================

def normalize_input(value):
    """
    Cleans the user's search text.
    """

    value = str(value).strip()

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value


search_input = normalize_input(
    user_input
)


# =========================================================
# SEARCH YAHOO FINANCE
# =========================================================

@st.cache_data(ttl=3600)
def search_yahoo_stocks(search_text):
    """
    Search Yahoo Finance for stocks.
    """

    if not search_text:
        return []

    try:

        url = (
            "https://query1.finance.yahoo.com/v1/"
            "finance/search"
        )

        params = {
            "q": search_text,
            "quotesCount": 20,
            "newsCount": 0
        }

        response = requests.get(
            url,
            params=params,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        result = response.json()

        quotes = result.get(
            "quotes",
            []
        )

        stocks = []

        for quote in quotes:

            if quote.get(
                "quoteType"
            ) != "EQUITY":

                continue

            symbol = quote.get(
                "symbol"
            )

            name = (
                quote.get("longname")
                or quote.get("shortname")
                or symbol
            )

            exchange = (
                quote.get("exchange")
                or quote.get("fullExchangeName")
                or ""
            )

            if not symbol:
                continue

            stocks.append(
                {
                    "symbol": symbol.upper(),
                    "name": name,
                    "exchange": exchange
                }
            )

        return stocks

    except Exception as e:
        st.sidebar.error(f"Stock search error: {e}")
        return []

# =========================================================
# STOCK SEARCH
# =========================================================

ticker = None

selected_stock_name = None
selected_exchange = None


if search_input:

    if len(search_input) < 2:

        st.sidebar.caption(
            "Type at least 2 characters to search."
        )

    else:

        search_results = search_yahoo_stocks(
            search_input
        )

        if search_results:

            result_labels = []

            for stock in search_results:

                label = (
                    f"{stock['name']} "
                    f"— {stock['symbol']}"
                )

                if stock["exchange"]:

                    label += (
                        f" — {stock['exchange']}"
                    )

                result_labels.append(
                    label
                )

            selected_index = st.sidebar.selectbox(
                "Matching stocks",
                range(
                    len(result_labels)
                ),
                format_func=lambda i:
                    result_labels[i]
            )

            selected_stock = (
                search_results[
                    selected_index
                ]
            )

            ticker = selected_stock[
                "symbol"
            ]

            selected_stock_name = (
                selected_stock[
                    "name"
                ]
            )

            selected_exchange = (
                selected_stock[
                    "exchange"
                ]
            )

            st.sidebar.success(
                f"Analyzing: {ticker}"
            )

        else:

            st.sidebar.warning(
                "No matching stocks were found. "
                "Try the company name or ticker."
            )


# =========================================================
# STOP IF NO STOCK SELECTED
# =========================================================

if not ticker:

    st.info(
        "👈 Search for a stock from the sidebar "
        "to start the analysis."
    )

    st.stop()


# =========================================================
# DOWNLOAD STOCK DATA
# =========================================================

@st.cache_data(ttl=900)
def download_data(
    ticker,
    period
):

    try:

        data = yf.download(
            ticker,
            period=period,
            auto_adjust=False,
            progress=False
        )

        return data

    except Exception:

        return pd.DataFrame()


data = download_data(
    ticker,
    period
)


# =========================================================
# CHECK DATA
# =========================================================

if data.empty:

    st.error(
        f"""
        No market data was found for **{ticker}**.

        Yahoo Finance may not have historical
        market data available for this security.

        Try another result from the search.
        """
    )

    st.stop()


# =========================================================
# HANDLE MULTIINDEX
# =========================================================

if isinstance(
    data.columns,
    pd.MultiIndex
):

    data.columns = (
        data.columns
        .get_level_values(0)
    )


# =========================================================
# REQUIRED COLUMNS
# =========================================================

required_columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:

    st.error(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )

    st.stop()


# =========================================================
# CLEAN DATA
# =========================================================

data = data.copy()

data = data.dropna(
    subset=[
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]
)


# =========================================================
# DAILY RETURN
# =========================================================

data["Daily_Return"] = (
    data["Close"]
    .pct_change()
)


# =========================================================
# MOVING AVERAGES
# =========================================================

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


# =========================================================
# VOLUME ANALYSIS
# =========================================================

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


# =========================================================
# ROLLING VOLATILITY
# =========================================================

data["Rolling_Volatility"] = (
    data["Daily_Return"]
    .rolling(20)
    .std()
)


# =========================================================
# RETURN Z-SCORE
# =========================================================

return_mean = (
    data["Daily_Return"]
    .rolling(20)
    .mean()
)

return_std = (
    data["Daily_Return"]
    .rolling(20)
    .std()
)

data["Return_Z"] = (
    (
        data["Daily_Return"]
        - return_mean
    )
    /
    return_std.replace(
        0,
        np.nan
    )
)


# =========================================================
# VOLUME Z-SCORE
# =========================================================

volume_mean = (
    data["Volume"]
    .rolling(20)
    .mean()
)

volume_std = (
    data["Volume"]
    .rolling(20)
    .std()
)

data["Volume_Z"] = (
    (
        data["Volume"]
        - volume_mean
    )
    /
    volume_std.replace(
        0,
        np.nan
    )
)


# =========================================================
# ANOMALY SCORE
# =========================================================

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


# =========================================================
# CLASSIFICATION
# =========================================================

def classify_anomaly(score):

    if score >= 75:

        return "Extreme"

    elif score >= 50:

        return "Unusual"

    else:

        return "Normal"


data["Classification"] = (
    data["Anomaly_Score"]
    .apply(
        classify_anomaly
    )
)


# =========================================================
# LATEST DATA
# =========================================================

latest = data.iloc[-1]

latest_price = float(
    latest["Close"]
)

latest_return = float(
    latest["Daily_Return"]
)

latest_anomaly = float(
    latest["Anomaly_Score"]
)

latest_return_z = float(
    latest["Return_Z"]
)

latest_volume_z = float(
    latest["Volume_Z"]
)

latest_volume_ratio = float(
    latest["Volume_Ratio"]
)

latest_classification = (
    latest["Classification"]
)


# =========================================================
# FIND MAJOR ANOMALY
# =========================================================

valid_data = data.dropna(
    subset=[
        "Anomaly_Score",
        "Daily_Return",
        "Return_Z",
        "Volume_Z",
        "Volume_Ratio"
    ]
)


if valid_data.empty:

    st.error(
        "Not enough data to calculate anomalies."
    )

    st.stop()


major_event = (
    valid_data.loc[
        valid_data[
            "Anomaly_Score"
        ].idxmax()
    ]
)


major_event_date = (
    major_event.name
)

major_event_return = float(
    major_event["Daily_Return"]
)

major_event_anomaly = float(
    major_event["Anomaly_Score"]
)

major_event_return_z = float(
    major_event["Return_Z"]
)

major_event_volume_z = float(
    major_event["Volume_Z"]
)

major_event_volume_ratio = float(
    major_event["Volume_Ratio"]
)

major_event_classification = (
    major_event["Classification"]
)


# =========================================================
# HEADER
# =========================================================

st.divider()

st.subheader(
    f"📊 Finesight Analysis — {selected_stock_name}"
)

st.caption(
    f"Ticker: {ticker}"
)


# =========================================================
# CURRENT STOCK OVERVIEW
# =========================================================

st.markdown(
    "### Current Stock Overview"
)

col1, col2, col3, col4 = (
    st.columns(4)
)


with col1:

    st.metric(
        "Latest Price",
        f"{latest_price:,.2f}"
    )


with col2:

    st.metric(
        "Daily Return",
        f"{latest_return * 100:.2f}%"
    )


with col3:

    st.metric(
        "Anomaly Score",
        f"{latest_anomaly:.1f}/100"
    )


with col4:

    st.metric(
        "Classification",
        latest_classification
    )


# =========================================================
# ANOMALY DETAILS
# =========================================================

st.markdown(
    "### 🚨 Anomaly Analysis"
)

col1, col2, col3, col4 = (
    st.columns(4)
)


with col1:

    st.metric(
        "Return Z-Score",
        f"{latest_return_z:.2f}"
    )


with col2:

    st.metric(
        "Volume Z-Score",
        f"{latest_volume_z:.2f}"
    )


with col3:

    st.metric(
        "Volume vs Normal",
        f"{latest_volume_ratio:.2f}×"
    )


with col4:

    st.metric(
        "Anomaly",
        latest_classification
    )


# =========================================================
# MAJOR ANOMALY
# =========================================================

st.divider()

st.subheader(
    "🔴 Major Anomaly Detected"
)

col1, col2, col3, col4 = (
    st.columns(4)
)


with col1:

    st.metric(
        "Date",
        pd.Timestamp(
            major_event_date
        ).strftime(
            "%d %b %Y"
        )
    )


with col2:

    st.metric(
        "Price Movement",
        f"{major_event_return * 100:.2f}%"
    )


with col3:

    st.metric(
        "Anomaly Score",
        f"{major_event_anomaly:.2f}/100"
    )


with col4:

    st.metric(
        "Classification",
        major_event_classification
    )


# =========================================================
# WHY FLAGGED
# =========================================================

st.markdown(
    "### 🔍 Why Was It Flagged?"
)

if abs(
    major_event_return_z
) >= 2:

    st.write(
        "• Price movement was significantly "
        "larger than normal."
    )

if abs(
    major_event_volume_z
) >= 2:

    st.write(
        "• Trading volume was significantly "
        "higher than normal."
    )

if major_event_return < 0:

    st.write(
        "• The movement was strongly negative."
    )

elif major_event_return > 0:

    st.write(
        "• The movement was strongly positive."
    )

if major_event_volume_ratio >= 2:

    st.write(
        f"• Trading volume was approximately "
        f"{major_event_volume_ratio:.2f}× "
        "the recent normal level."
    )


# =========================================================
# MARKET CONTEXT
# =========================================================

st.divider()

st.subheader(
    "🌐 Market Context"
)

market_return = None
relative_return = None


# =========================================================
# NIFTY ONLY FOR INDIAN NSE STOCKS
# =========================================================

if ticker.upper().endswith(".NS"):

    try:

        nifty = yf.download(
            "^NSEI",
            period=period,
            auto_adjust=False,
            progress=False
        )

        if isinstance(
            nifty.columns,
            pd.MultiIndex
        ):

            nifty.columns = (
                nifty.columns
                .get_level_values(0)
            )

        if not nifty.empty:

            nifty["Daily_Return"] = (
                nifty["Close"]
                .pct_change()
            )

            event_timestamp = (
                pd.Timestamp(
                    major_event_date
                )
            )

            available_dates = (
                nifty.index[
                    nifty.index
                    <= event_timestamp
                ]
            )

            if len(
                available_dates
            ) > 0:

                nearest_date = (
                    available_dates[-1]
                )

                market_return = float(
                    nifty.loc[
                        nearest_date,
                        "Daily_Return"
                    ]
                )

                relative_return = (
                    major_event_return
                    - market_return
                )

    except Exception:

        market_return = None


# =========================================================
# DISPLAY MARKET CONTEXT
# =========================================================

if market_return is not None:

    col1, col2, col3 = (
        st.columns(3)
    )

    with col1:

        st.metric(
            "Stock Return",
            f"{major_event_return * 100:.2f}%"
        )

    with col2:

        st.metric(
            "NIFTY Return",
            f"{market_return * 100:.2f}%"
        )

    with col3:

        st.metric(
            "Relative Performance",
            f"{relative_return * 100:.2f}%"
        )


    if abs(
        relative_return
    ) >= 0.04:

        if relative_return < 0:

            st.info(
                "The stock performed substantially "
                "worse than NIFTY around this event, "
                "suggesting the movement may have been "
                "relatively stock-specific."
            )

        else:

            st.info(
                "The stock performed substantially "
                "better than NIFTY around this event."
            )

    elif (
        major_event_return < 0
        and market_return < 0
    ):

        st.info(
            "Both the stock and broader market "
            "declined around the same time."
        )

    elif (
        major_event_return > 0
        and market_return > 0
    ):

        st.info(
            "Both the stock and broader market "
            "were positive around the same time."
        )

    else:

        st.info(
            "The stock and broader market showed "
            "different movements."
        )

else:

    if ticker.upper().endswith(".NS"):

        st.info(
            "NIFTY comparison could not be "
            "calculated for this event."
        )

    else:

        st.info(
            "NIFTY comparison is only used for "
            "Indian NSE stocks. International stocks "
            "are analyzed without NIFTY comparison."
        )


# =========================================================
# NEWS
# =========================================================

st.divider()

st.subheader(
    "📰 News Around Major Anomaly"
)

news_articles = get_news(
    ticker,
    major_event_date,
    days_before=2,
    days_after=2
)


# =========================================================
# NLP
# =========================================================

event_category = None
event_confidence = None


if news_articles:

    try:

        result = analyze_news(
            news_articles
        )

        if isinstance(
            result,
            dict
        ):

            event_category = (
                result.get(
                    "category"
                )
            )

            event_confidence = (
                result.get(
                    "confidence"
                )
            )

    except Exception:

        pass


# =========================================================
# NLP RESULT
# =========================================================

if event_category:

    st.markdown(
        f"**Possible News Category:** "
        f"{event_category}"
    )

    if event_confidence is not None:

        try:

            st.caption(
                f"NLP confidence: "
                f"{float(event_confidence):.0%}"
            )

        except Exception:

            pass

    st.caption(
        "This indicates a possible associated "
        "news category. It does not prove that "
        "the news caused the stock movement."
    )


# =========================================================
# SIMPLE NEWS SUMMARY
# =========================================================

def simple_news_summary(
    title,
    description
):

    if not description:

        return (
            "The available news feed does not "
            "provide a detailed description for "
            "this headline."
        )

    text = str(
        description
    ).strip()

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    if not text:

        return (
            "No additional summary was available "
            "from the news feed."
        )

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    summary = " ".join(
        sentences[:2]
    )

    if len(summary) > 350:

        summary = (
            summary[:347]
            .rsplit(
                " ",
                1
            )[0]
            + "..."
        )

    return summary


# =========================================================
# DISPLAY NEWS
# =========================================================

if news_articles:

    for article in news_articles[:8]:

        title = article.get(
            "title",
            "News headline"
        )

        source = article.get(
            "source",
            "Unknown source"
        )

        published = article.get(
            "published",
            ""
        )

        description = article.get(
            "description",
            ""
        )

        link = article.get(
            "link"
        )

        summary = simple_news_summary(
            title,
            description
        )

        st.markdown(
            f"### 📰 {title}"
        )

        st.write(
            f"**Simple summary:** {summary}"
        )

        source_text = ""

        if source:

            source_text += (
                f"Source: {source}"
            )

        if published:

            if source_text:

                source_text += " | "

            source_text += (
                published
            )

        if source_text:

            st.caption(
                source_text
            )

        if link:

            st.markdown(
                f"[Read full article →]({link})"
            )

        st.divider()

else:

    st.info(
        "No relevant news was found around "
        "the major anomaly."
    )


# =========================================================
# WHAT HAPPENED?
# =========================================================

st.subheader(
    "📌 What Happened?"
)


def generate_simple_summary(
    event_date,
    event_return,
    anomaly_score,
    classification,
    volume_ratio,
    stock_return,
    market_return=None,
    relative_return=None,
    event_category=None
):

    # -----------------------------------------------------
    # PRICE MOVEMENT
    # -----------------------------------------------------

    if event_return <= -0.05:

        movement = "fell sharply"

    elif event_return <= -0.02:

        movement = "fell noticeably"

    elif event_return < 0:

        movement = "fell slightly"

    elif event_return >= 0.05:

        movement = "rose sharply"

    elif event_return >= 0.02:

        movement = "rose noticeably"

    elif event_return > 0:

        movement = "rose slightly"

    else:

        movement = "barely moved"


    # -----------------------------------------------------
    # VOLUME
    # -----------------------------------------------------

    if np.isfinite(
        volume_ratio
    ):

        if volume_ratio >= 3:

            volume_text = (
                f"Trading activity was extremely "
                f"high, at about "
                f"{volume_ratio:.1f}× normal."
            )

        elif volume_ratio >= 2:

            volume_text = (
                f"Trading activity was much higher "
                f"than usual, at about "
                f"{volume_ratio:.1f}× normal."
            )

        elif volume_ratio >= 1.3:

            volume_text = (
                f"Trading activity was somewhat "
                f"higher than usual "
                f"({volume_ratio:.1f}× normal)."
            )

        elif volume_ratio < 0.7:

            volume_text = (
                "Trading activity was lower "
                "than usual."
            )

        else:

            volume_text = (
                "Trading activity was close "
                "to its normal level."
            )

    else:

        volume_text = (
            "Trading volume could not be "
            "compared with its recent average."
        )


    # -----------------------------------------------------
    # MARKET TEXT
    # -----------------------------------------------------

    market_text = ""

    if (
        market_return is not None
        and relative_return is not None
    ):

        if abs(
            relative_return
        ) >= 0.04:

            if relative_return < 0:

                market_text = (
                    "The stock performed much worse "
                    "than the broader market, so the "
                    "movement appears relatively "
                    "stock-specific."
                )

            else:

                market_text = (
                    "The stock performed much better "
                    "than the broader market."
                )

        elif (
            stock_return < 0
            and market_return < 0
        ):

            market_text = (
                "The broader market also declined "
                "around the same time."
            )

        elif (
            stock_return > 0
            and market_return > 0
        ):

            market_text = (
                "The broader market was also positive "
                "around the same time."
            )

        elif (
            stock_return < 0
            and market_return >= 0
        ):

            market_text = (
                "The broader market was stable or "
                "positive while this stock declined."
            )

        elif (
            stock_return > 0
            and market_return <= 0
        ):

            market_text = (
                "The broader market was weak while "
                "this stock gained."
            )


    # -----------------------------------------------------
    # NEWS CATEGORY
    # -----------------------------------------------------

    category_text = ""

    if event_category:

        category = str(
            event_category
        ).lower()

        if (
            "earning" in category
            or "result" in category
        ):

            category_text = (
                "News related to company earnings "
                "or financial results was identified "
                "around this period."
            )

        elif "corporate action" in category:

            category_text = (
                "A corporate action such as a "
                "dividend, bonus, split, or similar "
                "event was identified."
            )

        elif (
            "regulatory" in category
            or "legal" in category
        ):

            category_text = (
                "Regulatory or legal developments "
                "were reported around this period."
            )

        elif (
            "global tech" in category
            or "ai" in category
        ):

            category_text = (
                "Technology or AI-related developments "
                "were reported around this period."
            )

        elif (
            "sector" in category
            or "industry" in category
        ):

            category_text = (
                "Developments affecting the broader "
                "sector or industry were reported."
            )

        elif (
            "market" in category
            or "macro" in category
        ):

            category_text = (
                "Broader market or economic developments "
                "were reported around this period."
            )


    # -----------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------

    summary = (
        f"On "
        f"{pd.Timestamp(event_date).strftime('%d %b %Y')}, "
        f"the stock {movement} by "
        f"{abs(event_return) * 100:.2f}%. "
        f"Finesight detected this as a "
        f"{str(classification).lower()} movement "
        f"with an anomaly score of "
        f"{anomaly_score:.1f}/100. "
        f"{volume_text} "
    )

    if market_text:

        summary += (
            market_text
            + " "
        )

    if category_text:

        summary += category_text

    return summary.strip()


summary = generate_simple_summary(
    event_date=major_event_date,
    event_return=major_event_return,
    anomaly_score=major_event_anomaly,
    classification=major_event_classification,
    volume_ratio=major_event_volume_ratio,
    stock_return=major_event_return,
    market_return=market_return,
    relative_return=relative_return,
    event_category=event_category
)


st.info(
    summary
)


# =========================================================
# PRICE CHART
# =========================================================

st.divider()

st.subheader(
    "📈 Stock Price"
)

price_data = data[
    [
        "Close",
        "MA5",
        "MA20"
    ]
]

st.line_chart(
    price_data
)


# =========================================================
# VOLUME CHART
# =========================================================

st.subheader(
    "📊 Trading Volume"
)

volume_data = data[
    [
        "Volume",
        "Volume_Average_20"
    ]
]

st.line_chart(
    volume_data
)


# =========================================================
# ANOMALY CHART
# =========================================================

st.subheader(
    "🚨 Anomaly Score Over Time"
)

anomaly_data = data[
    [
        "Anomaly_Score"
    ]
]

st.line_chart(
    anomaly_data
)


# =========================================================
# MAJOR EVENTS
# =========================================================

st.divider()

st.subheader(
    "📋 Major Events"
)

top_events = (
    data[
        [
            "Close",
            "Daily_Return",
            "Volume_Ratio",
            "Return_Z",
            "Volume_Z",
            "Anomaly_Score",
            "Classification"
        ]
    ]
    .dropna(
        subset=[
            "Anomaly_Score"
        ]
    )
    .sort_values(
        "Anomaly_Score",
        ascending=False
    )
    .head(10)
    .copy()
)


if not top_events.empty:

    display_events = top_events.copy()

    display_events[
        "Daily_Return"
    ] = (
        display_events[
            "Daily_Return"
        ] * 100
    )

    display_events = (
        display_events.rename(
            columns={
                "Close": "Price",
                "Daily_Return": "Return (%)",
                "Volume_Ratio": "Volume Ratio",
                "Return_Z": "Return Z",
                "Volume_Z": "Volume Z",
                "Anomaly_Score": "Anomaly Score",
                "Classification": "Classification"
            }
        )
    )

    display_events.index = (
        pd.to_datetime(
            display_events.index
        ).strftime(
            "%d %b %Y"
        )
    )

    st.dataframe(
        display_events,
        width="stretch"
    )


# =========================================================
# NEXT-DAY ANOMALY PREDICTION
# =========================================================

st.divider()

st.markdown(
    "## 🔮 Next-Day Anomaly Prediction"
)

st.caption(
    "Estimated anomaly level for the next trading session "
    "using a Random Forest model trained on historical "
    "market behavior."
)

try:

    prediction_result = predict_next_anomaly(
        data
    )

    predicted_score = prediction_result[
        "score"
    ]

    predicted_classification = (
        prediction_result[
            "classification"
        ]
    )

    prediction_confidence = (
        prediction_result[
            "confidence"
        ]
    )

    # -----------------------------------------------------
    # PREDICTION METRICS
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Predicted Anomaly Score",
            f"{predicted_score:.1f}/100"
        )

    with col2:

        st.metric(
            "Predicted Classification",
            predicted_classification
        )

    with col3:

        st.metric(
            "Model Confidence",
            f"{prediction_confidence:.0f}%"
        )


    # -----------------------------------------------------
    # INTERPRETATION
    # -----------------------------------------------------

    if predicted_classification == "Extreme":

        st.warning(
            "The model estimates an unusually high "
            "anomaly level for the next trading session."
        )

    elif predicted_classification == "Unusual":

        st.info(
            "The model estimates a higher-than-normal "
            "anomaly level for the next trading session."
        )

    else:

        st.success(
            "The model estimates a relatively normal "
            "anomaly level for the next trading session."
        )


    # -----------------------------------------------------
    # DISCLAIMER
    # -----------------------------------------------------

    st.caption(
        "This is a statistical prediction of anomaly intensity, "
        "not a prediction of whether the stock price will rise "
        "or fall."
    )

except Exception as e:

    st.warning(
        f"Next-day anomaly prediction unavailable: {e}"
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Finesight provides statistical market analysis "
    "and does not provide financial advice or "
    "BUY/SELL recommendations."
)