import base64
import io

from flask import Flask, request, render_template
import yfinance
import json
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
from scrapper import fetch_recent_news_for_ticker

application = Flask(__name__)

def predict(ticker):
    pass

@application.route('/', methods=['GET', 'POST'])
def render_home_page_and_results():
    list_of_tickers = ['AAPL', 'GOOG', 'TSLA', 'AMZN', 'META']
    selected_ticker = request.form.get('selected_ticker')
    if request.method == 'POST':
        data = yfinance.Ticker(selected_ticker).history(period='1mo')
        close_prices = data['Close']

        # PLT code
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(data.index, close_prices, linewidth=3)
        ax.set_title(f'{selected_ticker} – Last 30 Days (Close Prices)', fontsize=12)
        ax.set_xlabel('Date', fontsize=8)
        ax.set_ylabel('Price $', fontsize=15)

        # Save to in-memory buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        # Encode as base64 string
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plot_url = f"data:image/png;base64,{img_base64}"

        html_table = data.to_html(classes='stock-table', border=1, justify='left')
        articles = fetch_recent_news_for_ticker(selected_ticker)
        return render_template("results.html", selected_ticker=selected_ticker, html_table=html_table, plot_url=plot_url, articles=articles)
    return render_template("homepage.html", list_of_tickers=list_of_tickers)

# @application.route('/resume', methods=['GET'])
# def render_resume():
#     return render_template("resume.html")
#
# @application.route('/other_projects', methods=['GET'])
# def render_other_projects():
#     return render_template("other_projects.html")

# @application.route('/DTSC_691_project', methods=['GET', 'POST'])
# def render_DTSC_691_project():
#     list_of_tickers = ['General_SP500', 'AAPL', 'GOOG', 'TSLA', 'AMZN', 'META']
#     selected_ticker = request.form.get('ticker')
#
#     return render_template("index.html", list_of_tickers=list_of_tickers, selected_ticker=selected_ticker)


if __name__ == "__main__":
    # Start the application
    # LOCAL:
    # application.run(host="localhost", port=5000)
    application.run(debug=True)
