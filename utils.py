import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


def generate_emoji(
    emoji_type,
    rotation=0,
    mouth_shift=(0, 0),
    face_radius_size=10,
    size=28,
    color=255,
    eye_size=2,
):
    size = size * 2

    # TODO: Add crop  so no black

    # Create a blank image
    img = Image.new("L", (size, size), color=color)  # Grayscale
    draw = ImageDraw.Draw(img)
    assert type(eye_size) == int, "eye_size must be an integer"
    assert (
        type(mouth_shift[0]) == int and type(mouth_shift[1]) == int
    ), "mouth_shift must be an integer"

    assert (
        type(face_radius_size) == int and face_radius_size < size
    ), "face_radius_size must be an integer"

    # Draw face (circle)
    face_radius = face_radius_size
    face_center = (size // 2, size // 2)
    draw.ellipse(
        [
            face_center[0] - face_radius,
            face_center[1] - face_radius,
            face_center[0] + face_radius,
            face_center[1] + face_radius,
        ],
        fill=200,
    )

    # Draw eyes
    eye_positions = [
        (face_center[0] - 5, face_center[1] - 5),
        (face_center[0] + 5, face_center[1] - 5),
    ]
    for eye in eye_positions:
        draw.ellipse(
            [
                eye[0] - eye_size,
                eye[1] - eye_size,
                eye[0] + eye_size,
                eye[1] + eye_size,
            ],
            fill=0,
        )

    # Draw mouth based on emoji type
    mouth_center = (face_center[0], face_center[1] + 5)
    mouth_center = (
        mouth_center[0] + mouth_shift[0],
        mouth_center[1] + mouth_shift[1],
    )
    if emoji_type == "sad":
        draw.arc(
            [
                mouth_center[0] - 6,
                mouth_center[1] - 2,
                mouth_center[0] + 6,
                mouth_center[1] + 4,
            ],
            190,
            350,
            fill=0,
            width=2,
        )
    elif emoji_type == "happy":
        draw.arc(
            [
                mouth_center[0] - 6,
                mouth_center[1] - 3,
                mouth_center[0] + 6,
                mouth_center[1] + 3,
            ],
            10,
            170,
            fill=0,
            width=2,
        )
    elif emoji_type == "neutral":
        draw.line(
            [
                mouth_center[0] - 6,
                mouth_center[1],
                mouth_center[0] + 6,
                mouth_center[1],
            ],
            fill=0,
            width=2,
        )

    # Rotate and convert to numpy array
    size = size // 2
    img = img.rotate(rotation)
    img = img.crop((size // 2, size // 2, size // 2 + size, size // 2 + size))
    img = np.array(img).astype(np.float32) / 255.0
    return img


def load_model(path="data/pretrained_model.pt"):
    model = torch.load(path)
    model.eval()
    return model


def visualize_attack(original, perturbed, perturbation, filename="outputs/attack.png"):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Original")

    plt.subplot(1, 3, 2)
    plt.imshow(perturbed, cmap="gray")
    plt.title("Perturbed")

    plt.subplot(1, 3, 3)
    plt.imshow(perturbation, cmap="RdBu", vmin=-0.1, vmax=0.1)
    plt.title("Perturbation")
    plt.colorbar()
    plt.savefig(filename)
    plt.close()


def calculate_stats(original, perturbed, true_label, pred_label):
    l2_dist = np.linalg.norm(original - perturbed)
    success = 1 if true_label != pred_label else 0
    return {"l2_distance": l2_dist, "success": success}


class EmojiDataset(Dataset):
    def __init__(self, data_dir, ratio=0.8, is_train=True):
        ratio = ratio if is_train else 1 - ratio
        self.images = np.load(f"{data_dir}/images.npy", allow_pickle=True)
        self.images = self.images[: int(len(self.images) * ratio)]
        self.labels = np.load(f"{data_dir}/labels.npy", allow_pickle=True)
        self.labels = self.labels[: int(len(self.labels) * ratio)]
        self.label_to_idx = {"happy": 0, "sad": 1, "neutral": 2}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0)
        label = self.label_to_idx[self.labels[idx]]
        return img, label
