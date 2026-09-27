import re


# =========================================================
# ROOT-CAUSE KEYWORDS
# =========================================================

CAUSE_KEYWORDS = {

    "Company Results / Earnings": [
        "earnings",
        "results",
        "quarterly results",
        "profit",
        "revenue",
        "net profit",
        "loss",
        "eps",
        "financial results",
        "quarterly earnings",
        "q1 results",
        "q2 results",
        "q3 results",
        "q4 results"
    ],

    "Corporate Action": [
        "dividend",
        "bonus",
        "stock split",
        "share split",
        "buyback",
        "merger",
        "acquisition",
        "demerger",
        "share issue"
    ],

    "Company Announcement": [
        "company announced",
        "company announces",
        "management",
        "new order",
        "order win",
        "contract",
        "partnership",
        "deal",
        "expansion",
        "investment",
        "new project"
    ],

    "Regulatory / Legal": [
        "sebi",
        "regulator",
        "regulatory",
        "court",
        "lawsuit",
        "legal",
        "penalty",
        "investigation",
        "notice",
        "fine",
        "ban"
    ],

    "Global Tech / AI": [
        "global tech selloff",
        "global tech sell-off",
        "tech selloff",
        "tech sell-off",
        "technology selloff",
        "technology sell-off",
        "ai tool",
        "ai impact",
        "ai fear",
        "artificial intelligence",
        "ai disruption",
        "ai concerns",
        "us tech stocks",
        "us technology stocks",
        "global technology"
    ],

    "Market / Macro": [
        "nifty",
        "sensex",
        "market",
        "rbi",
        "interest rate",
        "inflation",
        "gdp",
        "rupee",
        "global markets",
        "fed",
        "federal reserve",
        "tariff",
        "recession"
    ],

    "Sector / Industry": [
        "it sector",
        "banking sector",
        "technology sector",
        "auto sector",
        "pharma sector",
        "energy sector",
        "it stocks",
        "technology stocks",
        "banking stocks",
        "pharma stocks",
        "sector",
        "industry"
    ]
}


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# CLASSIFY ONE HEADLINE
# =========================================================

def classify_headline(headline):

    text = clean_text(headline)

    scores = {}

    for category, keywords in CAUSE_KEYWORDS.items():

        score = 0

        for keyword in keywords:

            keyword = clean_text(keyword)

            if keyword in text:

                score += 1

        scores[category] = score


    best_category = max(
        scores,
        key=scores.get
    )

    best_score = scores[
        best_category
    ]


    if best_score == 0:

        return {
            "category": "Unknown / Other",
            "score": 0
        }


    return {
        "category": best_category,
        "score": best_score
    }


# =========================================================
# ANALYZE ALL NEWS
# =========================================================

def analyze_news(articles):

    if not articles:

        return {
            "root_cause": "No relevant news found",
            "confidence": "Low",
            "articles": []
        }


    analyzed_articles = []

    category_counts = {}


    # -----------------------------------------------------
    # CLASSIFY EACH ARTICLE
    # -----------------------------------------------------

    for article in articles:

        headline = article.get(
            "title",
            ""
        )

        result = classify_headline(
            headline
        )

        category = result[
            "category"
        ]

        category_counts[
            category
        ] = category_counts.get(
            category,
            0
        ) + 1


        analyzed_articles.append({

            **article,

            "category": category,

            "keyword_score":
                result["score"]
        })


    # -----------------------------------------------------
    # REMOVE UNKNOWN ARTICLES
    # -----------------------------------------------------

    known_categories = {

        category: count

        for category, count
        in category_counts.items()

        if category != "Unknown / Other"
    }


    # -----------------------------------------------------
    # FIND DOMINANT CATEGORY
    # -----------------------------------------------------

    if known_categories:

        root_cause = max(
            known_categories,
            key=known_categories.get
        )

        highest_count = (
            known_categories[
                root_cause
            ]
        )

    else:

        root_cause = (
            "Unknown / Other"
        )

        highest_count = (
            category_counts.get(
                "Unknown / Other",
                0
            )
        )


    # -----------------------------------------------------
    # CONFIDENCE
    # -----------------------------------------------------

    total_articles = len(
        articles
    )

    confidence_ratio = (
        highest_count
        / total_articles
    )


    if confidence_ratio >= 0.60:

        confidence = "High"

    elif confidence_ratio >= 0.30:

        confidence = "Medium"

    else:

        confidence = "Low"


    return {

        "root_cause":
            root_cause,

        "confidence":
            confidence,

        "articles":
            analyzed_articles
    }