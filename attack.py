from utils import load_model, generate_emoji, visualize_attack, calculate_stats
import torch
import numpy as np


# TODO: Participants implement their attack here!
def create_adversarial_example(model, image, true_label, epsilon=0.1):
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    image_tensor.requires_grad = True

    output = model(image_tensor)
    loss = torch.nn.functional.cross_entropy(output, torch.tensor([true_label]))

    model.zero_grad()
    loss.backward()

    # FGSM attack
    perturbation = epsilon * image_tensor.grad.data.sign()
    perturbed_image = image_tensor + perturbation
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image.detach().numpy().squeeze()


def test_attack():
    model = load_model()
    emoji = generate_emoji("happy")
    perturbed = create_adversarial_example(model, emoji, 0)

    orig_pred = model(torch.tensor(emoji).unsqueeze(0).unsqueeze(0)).argmax().item()
    adv_pred = model(torch.tensor(perturbed).unsqueeze(0).unsqueeze(0)).argmax().item()

    visualize_attack(emoji, perturbed, perturbed - emoji)
    stats = calculate_stats(emoji, perturbed, 0, adv_pred)
    print(
        f"Attack success: {stats['success']}, L2 distance: {stats['l2_distance']:.4f}"
    )


if __name__ == "__main__":
    test_attack()
