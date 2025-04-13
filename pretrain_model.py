import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from model import EmojiClassifier
from utils import *

# Create output directory
os.makedirs("outputs/training", exist_ok=True)


def train_model():
    # Initialize model
    model = EmojiClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Create datasets
    train_dataset = EmojiDataset("./data/emojis", is_train=True)
    val_dataset = EmojiDataset("./data/emojis", is_train=False)

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Track metrics
    metrics = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    num_epochs = 5
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * X.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_train += (predicted == y).sum().item()
            total_train += y.size(0)

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for X, y in val_loader:
                outputs = model(X)
                loss = criterion(outputs, y)

                epoch_val_loss += loss.item() * X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct_val += (predicted == y).sum().item()
                total_val += y.size(0)

        # Calculate metrics
        train_loss = epoch_train_loss / total_train
        train_acc = correct_train / total_train
        val_loss = epoch_val_loss / total_val
        val_acc = correct_val / total_val

        # Store metrics
        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["train_acc"].append(train_acc)
        metrics["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")
        print("-" * 40)

    # Save model

    torch.save(model, "data/pretrained_model.pt")
    print("Saved pretrained model to data/pretrained_model.pt")

    # Plot training curves
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(metrics["train_loss"], label="Train Loss")
    plt.plot(metrics["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(metrics["train_acc"], label="Train Acc")
    plt.plot(metrics["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")

    plt.tight_layout()
    plt.savefig("outputs/training/training_metrics.png")
    plt.close()

    # Generate confusion matrix
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in val_loader:
            outputs = model(X)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Happy", "Sad", "Angry"]
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Validation Confusion Matrix")
    plt.savefig("outputs/training/confusion_matrix.png")
    plt.close()


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    train_model()
