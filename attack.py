from utils import load_model, generate_emoji, visualize_attack, calculate_stats
import torch
import numpy as np


# TODO: Participants implement their attack here!
def create_adversarial_example(model, image, true_label, epsilon=0.1):
    """
    Inputs:
    - model: Pretrained model
    - image: Original image (28, 28) numpy array
    - true_label: 0 (happy), 1 (sad), 2 (neutral)
    - epsilon: Perturbation strength

    Returns:
    - perturbed_image: Adversarial example (28, 28) numpy array
    """

    # TODO: Create an adversarial example
    perturbed_image = ...

    return perturbed_image.detach().numpy().squeeze()


def test_attack():
    model = load_model()
    emoji = generate_emoji("happy")
    perturbed = create_adversarial_example(model, emoji, 0)

    orig_pred = model(torch.tensor(emoji).unsqueeze(0)).argmax().item()
    adv_pred = model(torch.tensor(perturbed).unsqueeze(0)).argmax().item()

    visualize_attack(emoji, perturbed, perturbed - emoji)
    stats = calculate_stats(emoji, perturbed, 0, adv_pred)
    print(
        f"Attack success: {stats['success']}, L2 distance: {stats['l2_distance']:.4f}"
    )


if __name__ == "__main__":
    test_attack()
