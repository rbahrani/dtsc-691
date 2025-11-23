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

    # Keep original order & index
    df = data.copy().reset_index(drop=False)  # old index becomes a column named "index"
    df.rename(columns={"index": "_row_id"}, inplace=True)
    print("hello7, before", len(df))

    # Normalize dates
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()

    print("hello1")
    unique_tickers = df[ticker_col].dropna().unique().tolist()
    if len(unique_tickers) == 0 or df[date_col].isna().all():
        df["open_price"] = pd.NA
        df["close_price"] = pd.NA
        # restore original structure
        df = df.sort_values("_row_id").drop(columns=["_row_id"]).reset_index(drop=True)
        return df

    min_date = df[date_col].min()
    max_date = df[date_col].max()
    print("hello2", "min_date:", min_date, "max_date:", max_date)

    # Try to download; on any exception, just return df with None prices
    try:
        print("hello3, downloading with yfinance for", len(unique_tickers), "tickers")
        price_data = yf.download(
            unique_tickers,
            start=min_date,
            end=max_date + pd.Timedelta(days=1),
            group_by="ticker",
            auto_adjust=False,
            progress=False,
        )
    except Exception as e:
        print("yfinance error:", e)
        df["open_price"] = pd.NA
        df["close_price"] = pd.NA
        df = df.sort_values("_row_id").drop(columns=["_row_id"]).reset_index(drop=True)
        return df

    if price_data is None or price_data.empty:
        print("hello4 (empty price_data)")
        df["open_price"] = pd.NA
        df["close_price"] = pd.NA
        df = df.sort_values("_row_id").drop(columns=["_row_id"]).reset_index(drop=True)
        return df

    # ---- reshape price_data into [stock, date, open_price, close_price] ----
    if isinstance(price_data.columns, pd.MultiIndex):
        print("hello5 (multi-index)")

        # Keep only Open & Close
        try:
            price_oc = price_data.loc[:, (slice(None), ["Open", "Close"])]
        except KeyError:
            price_data_swapped = price_data.swaplevel(0, 1, axis=1)
            price_oc = price_data_swapped.loc[:, (slice(None), ["Open", "Close"])]

        # Stack tickers into rows
        stacked = price_oc.stack(level=0).reset_index()

        # First two columns are date + ticker, whatever they are called
        date_raw_col = stacked.columns[0]
        ticker_raw_col = stacked.columns[1]

        stacked.rename(
            columns={
                date_raw_col: date_col,
                ticker_raw_col: ticker_col,
                "Open": "open_price",
                "Close": "close_price",
            },
            inplace=True,
        )

    else:
        print("hello5 (single ticker)")
        stacked = price_data.reset_index()

        date_raw_col = stacked.columns[0]  # date index
        only_ticker = unique_tickers[0] if len(unique_tickers) > 0 else None
        stacked[ticker_col] = only_ticker

        stacked.rename(
            columns={
                date_raw_col: date_col,
                "Open": "open_price",
                "Close": "close_price",
            },
            inplace=True,
        )

    # Normalize date again
    stacked[date_col] = pd.to_datetime(stacked[date_col]).dt.normalize()

    # Only keep the columns we care about
    stacked = stacked[[ticker_col, date_col, "open_price", "close_price"]]

    # DEBUG: sizes
    print("price rows:", len(stacked), "input rows:", len(df))

    # Left join: this CANNOT create extra rows.
    merged = df.merge(stacked, on=[ticker_col, date_col], how="left")

    # Restore original row order and drop helper column
    merged = merged.sort_values("_row_id").drop(columns=["_row_id"]).reset_index(drop=True)

    print("hello6, merged rows:", len(merged))
    return merged
    # ticker_col = "stock"
    # date_col = "date"
    # df = data.copy()
    # df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    #
    # print("hello1")
    # unique_tickers = df[ticker_col].dropna().unique().tolist()
    # if len(unique_tickers) == 0 or df[date_col].isna().all():
    #     return df.assign(open_price=None, close_price=None)
    #
    # min_date = df[date_col].min()
    # max_date = df[date_col].max()
    # print("hello2")
    #
    # # Try to download; on any exception, just return df with None prices
    # try:
    #     print("hello3")
    #     price_data = yf.download(
    #         unique_tickers,
    #         start=min_date,
    #         end=max_date + pd.Timedelta(days=1),
    #         group_by="ticker",
    #         auto_adjust=False,
    #         progress=False,
    #     )
    # except Exception:
    #     df["open_price"] = None
    #     df["close_price"] = None
    #     return df
    #
    # if price_data is None or price_data.empty:
    #     print("hello4")
    #     df["open_price"] = None
    #     df["close_price"] = None
    #     return df
    #
    # # --- NEW: only keep Open & Close to simplify ---
    # if isinstance(price_data.columns, pd.MultiIndex):
    #     print("hello5 (multi-index)")
    #     # price_data columns look like: (ticker, field) or (field, ticker)
    #     # We only care about Open / Close
    #     try:
    #         price_oc = price_data.loc[:, (slice(None), ["Open", "Close"])]
    #     except KeyError:
    #         # yfinance sometimes flips level order; swap if needed
    #         price_data = price_data.swaplevel(0, 1, axis=1)
    #         price_oc = price_data.loc[:, (slice(None), ["Open", "Close"])]
    #
    #     # After this, index is Date, columns MultiIndex -> stack tickers
    #     stacked = price_oc.stack(level=0).reset_index()
    #
    #     # First two columns are date + ticker, regardless of their auto names
    #     date_raw_col = stacked.columns[0]
    #     ticker_raw_col = stacked.columns[1]
    #
    #     stacked.rename(
    #         columns={
    #             date_raw_col: date_col,
    #             ticker_raw_col: ticker_col,
    #             "Open": "open_price",
    #             "Close": "close_price",
    #         },
    #         inplace=True,
    #     )
    #
    # else:
    #     print("hello5 (single ticker)")
    #     # Single ticker -> regular columns: Date, Open, High, Low, Close, ...
    #     stacked = price_data.reset_index()
    #
    #     only_ticker = unique_tickers[0] if len(unique_tickers) > 0 else None
    #     stacked[ticker_col] = only_ticker
    #
    #     # First column is the date index
    #     date_raw_col = stacked.columns[0]
    #
    #     stacked.rename(
    #         columns={
    #             date_raw_col: date_col,
    #             "Open": "open_price",
    #             "Close": "close_price",
    #         },
    #         inplace=True,
    #     )
    #
    # # Normalize date & keep only the columns we want
    # stacked[date_col] = pd.to_datetime(stacked[date_col]).dt.normalize()
    #
    # keep_cols = [ticker_col, date_col, "open_price", "close_price"]
    # stacked = stacked[keep_cols]
    #
    # merged = df.merge(stacked, on=[ticker_col, date_col], how="left")
    # print("hello6")
    # print(merged)
    return merged


def calculate_daily_returns(data):
    return data.assign(daily_return=lambda df: df["close_price"] / df["open_price"] - 1)

def save_df_to_csv(df, filename):
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    stocks_df = pd.read_csv('C:/Users/rosie/dtsc-691/data/stocks.csv', encoding='utf-8')
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

    save_df_to_csv(stocks_df, 'C:/Users/rosie/dtsc-691/data/stocks_processed.csv')
