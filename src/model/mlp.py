# source_domain_diagnosis.py
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm   # ✅ import tqdm


# ===========================
# Define MLP model
# ===========================
class MLP(nn.Module):
    def __init__(self, in_dim, h1, h2, num_classes, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def evaluate(loader, model, device, criterion):
    model.eval()
    ys, ypreds, losses = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            losses.append(loss.item() * xb.size(0))
            ys.append(yb.cpu().numpy())
            ypreds.append(logits.argmax(dim=1).cpu().numpy())
    ys = np.concatenate(ys)
    ypreds = np.concatenate(ypreds)
    loss = np.sum(losses) / len(loader.dataset)
    acc = accuracy_score(ys, ypreds)
    return loss, acc, ys, ypreds


def main(args):
    # 1. Load data
    df = pd.read_csv(args.data)
    if "cls" not in df.columns:
        raise KeyError("'cls' column not found. Please make sure the CSV file contains a class label column named 'cls'")

    # Drop unused columns
    drop_cols = ["cls", "File Name"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["cls"]

    # 2. Handle missing values & encoding
    for col in X.columns:
        if X[col].dtype.kind in "biufc":
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna(X[col].mode().iloc[0])
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # Encode labels
    le_y = LabelEncoder()
    y_encoded = le_y.fit_transform(y.astype(str))
    classes = le_y.classes_

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values.astype(float))

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=args.test_size, random_state=args.seed, stratify=y_encoded
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=256, shuffle=False)

    # 4. Build MLP
    input_dim = X_train_t.shape[1]
    hidden1 = min(256, max(32, input_dim * 4))
    hidden2 = min(128, max(16, input_dim * 2))
    model = MLP(input_dim, hidden1, hidden2, len(classes), dropout=args.dropout).to(device)
    print(model)

    # 5. Training with tqdm
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []

    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
        model.train()
        epoch_losses = []

        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item() * xb.size(0))

        # Evaluate after each epoch
        train_loss, train_acc, _, _ = evaluate(train_loader, model, device, criterion)
        val_loss, val_acc, _, _ = evaluate(test_loader, model, device, criterion)

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

        if epoch % 50 == 0 or epoch == args.epochs:
            tqdm.write(f"Epoch {epoch}/{args.epochs}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

    # 6. Final evaluation on test set
    test_loss, test_acc, y_true, y_pred = evaluate(test_loader, model, device, criterion)
    print("\nFinal test accuracy:", test_acc)
    print("Classification report:\n", classification_report(y_true, y_pred, target_names=classes))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

    # 7. Save model and evaluation results
    os.makedirs(args.outputs, exist_ok=True)
    model_path = os.path.join(args.outputs, "mlp_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "classes": classes.tolist(),
        "input_columns": X.columns.tolist()
    }, model_path)

    summary = pd.DataFrame([{
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "num_classes": len(classes),
        "train_size": len(X_train),
        "test_size": len(X_test)
    }])
    summary_path = os.path.join(args.outputs, "mlp_eval_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"\nModel saved to: {model_path}")
    print(f"Evaluation summary saved to: {summary_path}")

    # 8. Plotting (optional)
    if args.plot:
        os.makedirs(args.outputs, exist_ok=True)

        # Loss plot
        plt.figure()
        plt.plot(range(1, args.epochs + 1), train_loss_hist, label="train_loss")
        plt.plot(range(1, args.epochs + 1), val_loss_hist, label="val_loss")
        plt.legend()
        plt.title("Loss")
        loss_plot_path = os.path.join(args.outputs, "loss_curve.png")
        plt.savefig(loss_plot_path)
        plt.close()
        print(f"Loss curve saved to: {loss_plot_path}")

        # Accuracy plot
        plt.figure()
        plt.plot(range(1, args.epochs + 1), train_acc_hist, label="train_acc")
        plt.plot(range(1, args.epochs + 1), val_acc_hist, label="val_acc")
        plt.legend()
        plt.title("Accuracy")
        acc_plot_path = os.path.join(args.outputs, "accuracy_curve.png")
        plt.savefig(acc_plot_path)
        plt.close()
        print(f"Accuracy curve saved to: {acc_plot_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Source domain fault diagnosis with MLP")
    parser.add_argument("--data", type=str, default="feature.csv", help="Path to input CSV data")
    parser.add_argument("--outputs", type=str, default="results", help="Directory to save model and results")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of test set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--plot", action="store_true", help="Plot training curves")
    args = parser.parse_args()

    main(args)
