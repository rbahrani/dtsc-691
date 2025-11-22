from flask import Flask, request, render_template
import pickle
from datetime import datetime

application = Flask(__name__)

def predict(ticker):
    pass

@application.route('/Home Page', methods=['GET'])
def render_home_page():
    return render_template("homepage.html")

@application.route('/Resume', methods=['GET'])
def render_resume():
    return render_template("resume.html")

@application.route('/Other Projects', methods=['GET'])
def render_other_projects():
    return render_template("other_projects.html")

@application.route('/DTSC-691 Project', methods=['GET', 'POST'])
def render_DTSC_691_project():
    list_of_tickers = ['General_SP500', 'AAPL', 'GOOG', 'TSLA', 'AMZN', 'META']
    selected_ticker = request.form.get('ticker')
    date_str = request.form.get('date')
    # date = datetime.strptime(date_str, '%Y-%m-%d').date()

    # process the data for the selected ticker
    predict(selected_ticker)

    return render_template("index.html", list_of_tickers=list_of_tickers, selected_ticker=selected_ticker)





if __name__ == "__main__":
    # Start the application
    application.run(host="localhost", port=5000)
