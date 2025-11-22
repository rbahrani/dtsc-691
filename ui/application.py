from flask import Flask, request, render_template
import pickle
from datetime import datetime

application = Flask(__name__)

def predict(ticker):
    pass

@application.route('/', methods=['GET', 'POST'])
def render_home_page():
    list_of_tickers = ['AAPL', 'GOOG', 'TSLA', 'AMZN', 'META']
    selected_ticker = request.form.get('ticker')
    if request.method == 'POST':
        print(selected_ticker)
        return render_template("results.html")
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
    application.run(host="localhost", port=5000)
