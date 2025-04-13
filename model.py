import torch
import torch.nn as nn


class EmojiClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3),  # 28x28 -> 26x26
            nn.ReLU(),
            nn.MaxPool2d(2),  # 26x26 -> 13x13
            nn.Conv2d(4, 8, kernel_size=3),  # 13x13 -> 11x11
            nn.ReLU(),
            nn.MaxPool2d(2),  # 11x11 -> 5x5
            nn.Flatten(),
            nn.Linear(8 * 5 * 5, num_classes),
        )

    def forward(self, x):
        return self.layers(x)


# Example usage
model = EmojiClassifier()
