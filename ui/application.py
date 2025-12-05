import base64
import io

from flask import Flask, request, render_template
import yfinance
import json
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from .scrapper import fetch_recent_news_for_ticker


class FinBERTRegressor(nn.Module):
    def __init__(self, model_name="ProsusAI/finbert", dropout=0.1):
        super().__init__()

        # backbone name must be "finbert" to match checkpoint keys
        self.finbert = AutoModel.from_pretrained(model_name)

        for p in self.finbert.parameters():
            p.requires_grad = False

        hidden_size = self.finbert.config.hidden_size  # 768 for FinBERT

        # From the error: reg_head.1.weight is [128, 770]
        # => input dimension to first Linear = 770 = 768 + 2 extra features
        self.extra_dim = 770 - hidden_size  # 2

        self.reg_head = nn.Sequential(
            nn.Dropout(dropout),                   # reg_head.0 (no params)
            nn.Linear(hidden_size + self.extra_dim, 128),  # reg_head.1
            nn.ReLU(),                             # reg_head.2
            nn.Linear(128, 1),                     # reg_head.3
        )

    def forward(self, input_ids, attention_mask, extra_features=None):
        """
        extra_features: tensor of shape (batch_size, extra_dim) if you have
                        numeric features (e.g., prices) like in training.
                        For now we can default to zeros.
        """
        outputs = self.finbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # (batch, 768)

        batch_size = cls_emb.size(0)

        if extra_features is None:
            # For now: fake the extra features as zeros, so we match the
            # trained layer shape and can use the checkpoint
            extra_features = torch.zeros(
                batch_size,
                self.extra_dim,
                device=cls_emb.device,
                dtype=cls_emb.dtype,
            )

        # (batch, 768 + 2) = (batch, 770)
        x = torch.cat([cls_emb, extra_features], dim=1)

        out = self.reg_head(x)
        return out.squeeze(-1)

app = Flask(__name__)
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FinBERTRegressor()
state_dict = torch.load("finbert_regressor_best.pt", map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

def predict_returns_for_headlines(headlines, max_length=64):
    """
    headlines: list[str], e.g. length 100
    returns: np.ndarray of shape (len(headlines),)
    """
    # Tokenize in one batch

    enc = tokenizer(
        headlines,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        preds = model(input_ids=input_ids, attention_mask=attention_mask)

    preds = preds.detach().cpu().numpy()
    return preds  # vector of predicted returns

def compute_stats(predictions):
    avg_return = float(np.mean(predictions))
    q05, q50, q95 = np.quantile(predictions, [0.05, 0.5, 0.95])
    return {
        "avg": avg_return,
        "q05": float(q05),
        "q50": float(q50),
        "q95": float(q95),
    }

import matplotlib
matplotlib.use("Agg")  # for server-side rendering
import matplotlib.pyplot as plt
import io, base64

def make_distribution_plot(preds):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.hist(preds, bins=300, alpha=0.7)
    ax.set_xlabel("Predicted returns")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Predicted Returns")

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)

    return f"data:image/png;base64,{image_base64}"

@app.route('/', methods=['GET', 'POST'])
def render_home_page_and_results():
    list_of_tickers = ['AAPL', 'GOOG', 'TSLA', 'AMZN', 'META']
    selected_ticker = request.form.get('selected_ticker')
    if request.method == 'POST':
        data = yfinance.Ticker(selected_ticker).history(period='1mo')
        close_prices = data['Close']

        # PLT code
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(15, 10))
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
        plot_url_30_days = f"data:image/png;base64,{img_base64}"

        html_table = data.to_html(classes='stock-table', border=1, justify='left')
        articles = fetch_recent_news_for_ticker(selected_ticker)
        headlines = [a["title"] for a in articles]  # extract just the text
        predictions = predict_returns_for_headlines(headlines)
        # print("PREDICTIONS ARE:")
        # print(predictions)
        stats = compute_stats(predictions)
        plot_url = make_distribution_plot(predictions)
        return render_template("results.html", selected_ticker=selected_ticker, html_table=html_table, plot_url_30_days=plot_url_30_days, plot_url=plot_url, articles=articles, predictions=predictions, avg_return=stats["avg"], q05=stats["q05"], q50=stats["q50"], q95=stats["q95"])
    return render_template("dtsc-691.html", list_of_tickers=list_of_tickers)

@app.route('/resume', methods=['GET'])
def render_resume():
    return render_template("resume.html")

@app.route('/biographical_homepage', methods=['GET'])
def render_biographical_homepage():
    return render_template("biographical_homepage.html")
@app.route('/other_projects', methods=['GET'])
def render_other_projects():
    return render_template("other_projects.html")


if __name__ == "__main__":
    # Start the app
    # LOCAL:
    app.run(host="localhost", port=5000)
    # app.run(debug=True)
