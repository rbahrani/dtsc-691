import urllib.parse
from datetime import datetime, timedelta, timezone

import feedparser
import requests


def fetch_recent_news_for_ticker(ticker: str, days: int = 5, max_items: int = 50):
    """
    Fetch recent news articles for a stock ticker from Google News RSS.

    Args:
        ticker: Stock ticker, e.g. "AAPL", "TSLA".
        days: Only keep articles from the last `days` days.
        max_items: Maximum number of RSS entries to inspect.

    Returns:
        List of dicts: [{title, link, published, source}, ...]
    """
    # Build a Google News RSS search query
    # We search for "<TICKER> stock" and restrict to last `days` days using when:Xd
    query = urllib.parse.quote_plus(f'"{ticker}" stock')
    rss_url = (
        f"https://news.google.com/rss/search?"
        f"q={query}+when:{days}d&hl=en-US&gl=US&ceid=US:en"
    )

    # Optional: simple requests check (not strictly necessary, but nice for errors)
    response = requests.get(rss_url, timeout=10)
    response.raise_for_status()

    # Parse the RSS feed
    feed = feedparser.parse(response.text)

    # Compute time cutoff
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    articles = []
    for entry in feed.entries[:max_items]:
        # Try to parse the published/updated time
        pub_dt = None
        if getattr(entry, "published_parsed", None):
            pub_dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        elif getattr(entry, "updated_parsed", None):
            pub_dt = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)

        # If we have a timestamp, enforce the cutoff
        if pub_dt and pub_dt < cutoff:
            continue

        # Try to get the source (publisher) name if present
        source_title = None
        if hasattr(entry, "source") and getattr(entry, "source", None):
            # feedparser often places source title as entry.source.title
            source_title = getattr(entry.source, "title", None)

        articles.append(
            {
                "title": entry.get("title"),
                "link": entry.get("link"),
                "published": pub_dt.isoformat() if pub_dt else None,
                "source": source_title,
            }
        )

    print(articles)
    print(len(articles))
    return articles

fetch_recent_news_for_ticker("AAPL", days=1, max_items=20)
