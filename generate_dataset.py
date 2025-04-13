from utils import generate_emoji
import matplotlib.pyplot as plt
import numpy as np
import os


def generate_large_dataset(save_dir="data/emojis", seed=42, n=10000):
    os.makedirs(save_dir, exist_ok=True)
    labels = []
    np.random.seed(seed)
    emojis = []

    for i in range(n):
        # Randomly choose type and rotation
        emoji_type = np.random.choice(["happy", "sad", "neutral"])
        rotation = np.random.uniform(-20, 20)
        mouth_shift = (np.random.randint(-2, 3), np.random.randint(-2, 3))
        face_radius_size = np.random.randint(8, 13)
        size = 28
        color = np.random.randint(240, 255)
        eye_size = np.random.randint(0, 3)
        random_emoji = generate_emoji(
            emoji_type,
            rotation=rotation,
            mouth_shift=mouth_shift,
            face_radius_size=face_radius_size,
            size=size,
            color=color,
            eye_size=np.random.randint(0, 3),
        )

        if i < 10:
            plt.figure(figsize=(16, 16))
            plt.imshow(random_emoji, cmap="gray")
            plt.title(
                f"Emoji {i}: {emoji_type}, Rotation: {rotation}\n, Mouth Shift: {mouth_shift}, Face Radius: {face_radius_size}\n, Color: {color}, Eye Size: {eye_size}"
            )
            plt.axis("off")
            plt.savefig(f"outputs/emoji_{i}.png")
            print(f"Emoji {i}: {emoji_type}, Rotation: {random_emoji}")

        emojis.append(random_emoji)
        # # Save as .npy file
        # np.save(f"{save_dir}/{i}.npy", random_emoji)
        labels.append(emoji_type)

    # Save labels
    np.save(f"{save_dir}/labels.npy", np.array(labels))
    np.save(f"{save_dir}/images.npy", np.array(emojis))


if __name__ == "__main__":
    generate_large_dataset(n=100000)
