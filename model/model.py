# ============================
# 1. Install & imports
# ============================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Optional: set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if device.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

# ============================
# 2. Data prep & 80/10/10 split
# ============================
# Assumed df columns:
# - "headline"           (text)
# - "stock"          (ticker string)
# - "open_price"     (float)
# - "close_price"    (float)
# - "target_return"  (float, what we want to predict)

TEXT_COL   = "headline"
TARGET_COL = "daily_return"
NUM_COLS   = ["open_price", "close_price"]
TICKER_COL = "stock"  # not used yet, but kept

df = pd.read_csv("C:/Users/rosie/dtsc-691/data/stocks_processed.csv", encoding="utf-8")

# Drop rows with missing values in key columns
df = df.dropna(subset=[TEXT_COL, TARGET_COL] + NUM_COLS).copy()

# Shuffle
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

# 80% train, 10% val, 10% test
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=SEED)
val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=SEED)

print("Train size:", len(train_df))
print("Val size  :", len(val_df))
print("Test size :", len(test_df))

# ============================
# 3. Scale numeric features (fit on train only)
# ============================
scaler = StandardScaler()
train_df[NUM_COLS] = scaler.fit_transform(train_df[NUM_COLS])
val_df[NUM_COLS]   = scaler.transform(val_df[NUM_COLS])
test_df[NUM_COLS]  = scaler.transform(test_df[NUM_COLS])

# ============================
# 4. Tokenizer & Dataset
# ============================
MODEL_NAME = "ProsusAI/finbert"  # or "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class NewsDataset(Dataset):
    def __init__(self, df, tokenizer, text_col, target_col, num_cols, max_len=128):
        self.texts   = df[text_col].astype(str).tolist()
        self.targets = df[target_col].astype(np.float32).values
        self.num_feats = df[num_cols].astype(np.float32).values  # shape [N, num_features]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text   = self.texts[idx]
        target = self.targets[idx]
        num_f  = self.num_feats[idx]

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"]    = torch.tensor(target, dtype=torch.float32)
        item["num_feats"] = torch.tensor(num_f, dtype=torch.float32)  # price features

        return item

# Create datasets
train_dataset = NewsDataset(train_df, tokenizer, TEXT_COL, TARGET_COL, NUM_COLS)
val_dataset   = NewsDataset(val_df,   tokenizer, TEXT_COL, TARGET_COL, NUM_COLS)
test_dataset  = NewsDataset(test_df,  tokenizer, TEXT_COL, TARGET_COL, NUM_COLS)

# DataLoaders
BATCH_SIZE = 16

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ============================
# 5. Model: FinBERT frozen + numeric features + regression head
# ============================
class FinBertRegressor(nn.Module):
    def __init__(self, base_model_name=MODEL_NAME, num_feature_dim=len(NUM_COLS), dropout=0.1):
        super().__init__()
        self.finbert = AutoModel.from_pretrained(base_model_name)

        # Freeze FinBERT weights
        for p in self.finbert.parameters():
            p.requires_grad = False

        hidden_size = self.finbert.config.hidden_size  # usually 768
        total_in = hidden_size + num_feature_dim       # CLS + numeric features

        self.reg_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_in, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask, num_feats, token_type_ids=None, labels=None):
        outputs = self.finbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # CLS token embedding
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        # Concatenate numeric features
        x = torch.cat([cls_emb, num_feats], dim=-1)   # [batch_size, hidden_size + num_dim]

        preds = self.reg_head(x).squeeze(-1)          # [batch_size]

        loss = None
        if labels is not None:
            loss = nn.MSELoss()(preds, labels)

        return preds, loss

model = FinBertRegressor().to(device)
print(model)

# ============================
# 6. Train / eval utilities
# ============================
EPOCHS = 5
LR = 2e-4

optimizer = AdamW(model.parameters(), lr=LR)

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        labels    = batch["labels"].to(device)
        num_feats = batch["num_feats"].to(device)

        optimizer.zero_grad()
        preds, loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            num_feats=num_feats,
            labels=labels
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            labels    = batch["labels"].to(device)
            num_feats = batch["num_feats"].to(device)

            preds, loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                num_feats=num_feats,
                labels=labels
            )

            total_loss += loss.item() * input_ids.size(0)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    mse = total_loss / len(dataloader.dataset)
    mae = mean_absolute_error(all_targets, all_preds)

    return mse, mae, all_preds, all_targets

# ============================
# 7. Training loop
# ============================
best_val_mae = float("inf")

for epoch in range(1, EPOCHS + 1):
    train_mse = train_one_epoch(model, train_loader, optimizer, device)
    val_mse, val_mae, _, _ = evaluate(model, val_loader, device)

    print(f"Epoch {epoch}/{EPOCHS}")
    print(f"  Train MSE: {train_mse:.6f}")
    print(f"  Val   MSE: {val_mse:.6f} | Val MAE: {val_mae:.6f}")

    # Simple "best model" tracking (by MAE)
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        torch.save(model.state_dict(), "finbert_regressor_best.pt")
        print("  --> Saved new best model")

# ============================
# 8. Final test evaluation
# ============================
# Load best model (optional but recommended)
model.load_state_dict(torch.load("finbert_regressor_best.pt", map_location=device))

test_mse, test_mae, test_preds, test_targets = evaluate(model, test_loader, device)

test_rmse = np.sqrt(test_mse)
test_r2   = r2_score(test_targets, test_preds)

print("\n=== Test set metrics ===")
print(f"MSE : {test_mse:.6f}")
print(f"RMSE: {test_rmse:.6f}")
print(f"MAE : {test_mae:.6f}")
print(f"R^2 : {test_r2:.6f}")