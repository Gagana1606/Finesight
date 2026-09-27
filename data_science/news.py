import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote
from datetime import datetime, timedelta
import re


def clean_text(text):

    if not text:
        return ""

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def get_news(
    ticker,
    event_date,
    days_before=2,
    days_after=2
):

    search_term = (
        ticker
        .replace(".NS", "")
        .replace(".BO", "")
    )

    if isinstance(
        event_date,
        str
    ):

        event_date = datetime.strptime(
            event_date,
            "%Y-%m-%d"
        )

    start_date = (
        event_date
        - timedelta(
            days=days_before
        )
    ).strftime(
        "%Y-%m-%d"
    )

    end_date = (
        event_date
        + timedelta(
            days=days_after
        )
    ).strftime(
        "%Y-%m-%d"
    )

    search_query = (
        f"{search_term} stock "
        f"after:{start_date} "
        f"before:{end_date}"
    )

    encoded_query = quote(
        search_query
    )

    url = (
        "https://news.google.com/rss/search?"
        f"q={encoded_query}"
        "&hl=en-IN"
        "&gl=IN"
        "&ceid=IN:en"
    )

    try:

        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        root = ET.fromstring(
            response.content
        )

        articles = []

        for item in root.findall(
            ".//item"
        ):

            title = item.findtext(
                "title"
            )

            link = item.findtext(
                "link"
            )

            published = item.findtext(
                "pubDate"
            )

            description = item.findtext(
                "description"
            )

            source_element = item.find(
                "source"
            )

            source = None

            if source_element is not None:

                source = (
                    source_element.text
                )

            if title:

                articles.append(
                    {
                        "title": title.strip(),
                        "link": link,
                        "published": published,
                        "source": source,
                        "description": clean_text(
                            description
                        )
                    }
                )

        return articles

    except Exception as e:

        print(
            f"News retrieval error: {e}"
        )

        return []