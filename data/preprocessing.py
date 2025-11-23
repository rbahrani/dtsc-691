import pandas as pd
import numpy as np
import yfinance as yf

# Note to self: Make sure that the dropna does it in place and also returns a df

def print_most_common_occurance(data):
    print(data["stock"].value_counts().head(100))

def drop_rows_with_less_than_n_occurences(data, n):
    stock_counts = data["stock"].value_counts()
    stocks_over_n = stock_counts[stock_counts >= n].index.tolist()
    filtered_df = data[data["stock"].isin(stocks_over_n)].copy()
    return filtered_df

def remove_rows_with_missing_values(data):
    return data.dropna()

def preprocess_headline_strings(data):
    data["headline"] = data["headline"].astype(str)
    data["headline"] = data["headline"].str.lower()
    data["headline"] = data["headline"].str.replace(r"[^0-9a-zA-Z ]+", " ", regex=True)
    data["headline"] = data["headline"].str.replace(r"\s+", " ", regex=True).str.strip()
    return data

def fix_date_formats(data):
    return data.assign(
        date=lambda df: pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    )

def augment_stocks_open_close_prices(data):
    ticker_col = "stock"
    date_col = "date"
    df = data.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()

    unique_tickers = df[ticker_col].dropna().unique().tolist()
    if len(unique_tickers) == 0 or df[date_col].isna().all():
        return df.assign(open_price=None, close_price=None)

    min_date = df[date_col].min()
    max_date = df[date_col].max()

    # Try to download; on any exception, just return df with None prices
    try:
        price_data = yf.download(
            unique_tickers,
            start=min_date,
            end=max_date + pd.Timedelta(days=1),
            group_by="ticker",
            auto_adjust=False,
            progress=False,
        )
    except Exception:
        df["open_price"] = None
        df["close_price"] = None
        return df

    if price_data is None or price_data.empty:
        df["open_price"] = None
        df["close_price"] = None
        return df

    # Reshape depending on whether we have multiple tickers (MultiIndex) or single
    if isinstance(price_data.columns, pd.MultiIndex):
        stacked = price_data.stack(level=1).reset_index()
        stacked.rename(
            columns={
                "level_1": ticker_col,
                "Open": "open_price",
                "Close": "close_price",
                "Date": date_col,
            },
            inplace=True,
        )
    else:
        stacked = price_data.reset_index()
        only_ticker = unique_tickers[0] if len(unique_tickers) > 0 else None
        stacked[ticker_col] = only_ticker
        stacked.rename(
            columns={
                "Open": "open_price",
                "Close": "close_price",
                "Date": date_col,
            },
            inplace=True,
        )

    stacked[date_col] = pd.to_datetime(stacked[date_col]).dt.normalize()
    stacked = stacked[[ticker_col, date_col, "open_price", "close_price"]]

    merged = df.merge(stacked, on=[ticker_col, date_col], how="left")
    return merged


def calculate_daily_returns(data):
    return data.assign(daily_return=lambda df: df["close"] / df["open"] - 1)

if __name__ == "__main__":
    stocks_df = pd.read_csv('stocks.csv', encoding='utf-8')
    print_most_common_occurance(stocks_df)
    print("test1")
    stocks_df = drop_rows_with_less_than_n_occurences(stocks_df, 1000)
    print("test2")
    stocks_df = remove_rows_with_missing_values(stocks_df)
    print("test3")
    stocks_df = preprocess_headline_strings(stocks_df)
    print("test4")
    stocks_df = fix_date_formats(stocks_df)
    print("test5")
    stocks_df = augment_stocks_open_close_prices(stocks_df)
    print("test6")
    stocks_df = calculate_daily_returns(stocks_df)
    print("test7")

    print(stocks_df)
