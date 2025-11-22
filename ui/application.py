from flask import Flask, request, render_template
import pickle
from datetime import datetime

application = Flask(__name__)

def predict(ticker):
    pass
@application.route('/', methods=['GET', 'POST'])
def template_ui():
    list_of_tickers = ['General_SP500', 'AAPL', 'GOOG', 'TSLA', 'AMZN', 'META']
    selected_ticker = request.form.get('ticker')
    date_str = request.form.get('date')
    #date = datetime.strptime(date_str, '%Y-%m-%d').date()

    # process the data for the selected ticker
    predict(selected_ticker)

    return render_template("index.html", list_of_tickers=list_of_tickers, selected_ticker=selected_ticker)


@application.route('/Home Page', methods=['GET'])

@application.route('/Resume', methods=['GET'])

@application.route('/Other Projects', methods=['GET'])

@application.route('/DTSC-691 Project', methods=['GET'])





if __name__ == "__main__":
    # Start the application
    application.run(host="localhost", port=5000)
