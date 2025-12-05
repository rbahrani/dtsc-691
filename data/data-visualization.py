import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator

df = pd.read_csv("stocks_processed_final.csv")
df_original = pd.read_csv("stocks.csv")

def show_daily_returns(df):

    # Filter out the outliers
    non_outliers = df[(df["daily_return"] <= 0.5)]
    non_outliers = non_outliers[(non_outliers["daily_return"] >= -0.5)]
    returns = non_outliers["daily_return"] * 100

    # Create the plot
    plt.figure(figsize=(12, 5))
    plt.title("Occurrences of Different Daily Returns")
    plt.xlabel("Stock Daily Return %")
    plt.ylabel("Occurrences")
    plt.hist(returns, bins=100)
    plt.grid(alpha=0.5)

    plt.show()
    plt.savefig("show_daily_returns.png")

def show_headline_per_stock(df):

    # Count occurrences per ticker
    ticker_counts = df["stock"].value_counts()
    top_10 = ticker_counts.iloc[:10]

    # Create the plot
    plt.figure(figsize=(12, 7))
    top_10.plot(kind="bar")
    plt.ylabel("Count of Headlines")
    plt.title("Dataset # of News Headlines for Each Stock")
    plt.xlabel("Stock")
    plt.grid(alpha=0.5, axis="y")

    # plt.show()
    plt.savefig("show_headline_per_stock.png")


def show_headline_volume_over_time(df):
    count_of_headlines_per_date = df.groupby("date").size()

    plt.figure(figsize=(10, 5))
    count_of_headlines_per_date.plot(kind="line")
    plt.title("Dataset's # of Headlines per Day")
    plt.xlabel("Date")
    plt.ylabel("Count of Headlines")
    plt.grid(alpha=0.3)
    # plt.show()
    plt.savefig("show_headline_volume_over_time.png")

def show_distribution_of_headline_length(df):
    str_lengths = df["headline"].str.len()
    plt.figure(figsize=(10, 7))
    sns.histplot(str_lengths, bins=100)
    plt.title("Distribution of Lengths of Headline Strings")
    plt.xlabel("Length of Headline Strings")
    plt.ylabel("Occurrences")

    # plt.show()
    plt.savefig("show_distribution_of_headline_length.png")


def get_statistical_analysis_on_raw_data(df):

    # Analysis of the news headlines

    df['headline_length'] = df['headline'].str.len()
    print(df['headline_length'].describe())

    # Median, Min and Max headline lengths
    print("Max headline length is: ", df['headline_length'].median())
    print("Max headline length is: ", df['headline_length'].max())
    print("Min headline length is: ", df['headline_length'].min())


def get_statistical_analysis_on_processed_data(df):
    daily_returns = df["daily_return"]
    print(daily_returns.describe())
    print(daily_returns.info())

    # Analyzing the min and max daily returns
    print("Max daily return is: ", daily_returns.max())
    print("Min daily return is: ", daily_returns.min())

    # Analysing the median
    print("Median daily return is: ", daily_returns.median())

    # Analysing the open prices of the stocks
    open_prices = df["open_price"]
    print(open_prices.describe())
    print(open_prices.info())

    # Analysing min, max, and median of the open prices
    print("Max open price is: ", open_prices.max())
    print("Min open price is: ", open_prices.min())
    print("Median open price is: ", open_prices.median())

    # Analysing the close prices of the stocks
    close_prices = df["close_price"]
    print(close_prices.describe())
    print(close_prices.info())

    # Analysing min, max, and median of the close prices
    print("Max close price is: ", close_prices.max())
    print("Min close price is: ", close_prices.min())
    print("Median close price is: ", close_prices.median())


if __name__ == "__main__":
    show_daily_returns(df)
    show_headline_per_stock(df)
    show_headline_volume_over_time(df)
    show_distribution_of_headline_length(df_original)
    get_statistical_analysis_on_raw_data(df_original)
    get_statistical_analysis_on_processed_data(df)