import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("stocks_processed_final.csv")
df_original = pd.read_csv("stocks.csv")

def show_daily_returns(df):
    plt.figure(figsize=(8, 5))
    plt.hist(df["daily_return"].dropna(), bins=50, edgecolor="black", alpha=0.7)
    plt.title("Distribution of Daily Returns")
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def show_headline_per_stock(df):
    # Count samples per ticker
    ticker_counts = df["date"].value_counts()

    # Optionally show only top N (e.g., top 20)
    TOP_N = 20
    top_ticker_counts = ticker_counts.head(TOP_N)

    plt.figure(figsize=(10, 6))
    top_ticker_counts.plot(kind="bar")
    plt.title(f"Number of Headlines per Stock (Top {TOP_N})")
    plt.xlabel("Stock")
    plt.ylabel("Number of Headlines")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("Total unique tickers:", df["date"].nunique())
    print(ticker_counts.describe())


def show_headline_volume_over_time(df):
    headline_per_day = df.groupby("date").size()

    plt.figure(figsize=(10, 5))
    headline_per_day.plot()
    plt.title("Number of Headlines per Day")
    plt.xlabel("Date")
    plt.ylabel("Number of Headlines")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def show_distribution_of_headline_length(df):
    headline_lengths = df["headline"].str.len()
    plt.figure(figsize=(10, 5))
    sns.histplot(headline_lengths, bins=50, kde=True)
    plt.title("Distribution of Headline Lengths")
    plt.xlabel("Headline Length")
    plt.ylabel("Frequency")


if __name__ == "__main__":
    show_daily_returns(df)
    show_headline_per_stock(df)
    show_headline_volume_over_time(df)
    show_distribution_of_headline_length(df_original)