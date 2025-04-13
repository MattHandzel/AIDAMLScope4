from utils import load_model
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class EmojiDataset(Dataset):
    def __init__(self, data_dir):
        self.images = np.load(f"{data_dir}/images.npy", allow_pickle=True)
        self.labels = np.load(f"{data_dir}/labels.npy", allow_pickle=True)
        self.label_to_idx = {"happy": 0, "sad": 1, "neutral": 2}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0)
        label = self.label_to_idx[self.labels[idx]]
        return img, label


def train():
    # Model and dataset
    model = load_model()
    dataset = EmojiDataset("data/emojis")
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(10):
        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Save model
    torch.save(model.state_dict(), "data/pretrained_model.pt")


if __name__ == "__main__":
    train()
