import torch
import torch.nn as nn


class EmojiClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.input_layer = nn.Conv2d(1, 4, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * 5 * 5, num_classes)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# Example usage
model = EmojiClassifier()
