from utils import *


# Complete FGSM implementation
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


# Test 100 examples and show statistics
def evaluate_attack():
    model = load_model()
    dataset = EmojiDataset("data/emojis")
    successes = []
    distances = []

    for i in range(100):
        img, label = dataset[i]
        perturbed = create_adversarial_example(model, img.squeeze().numpy(), label)

        pred = model(torch.tensor(perturbed).unsqueeze(0)).argmax().item()
        stats = calculate_stats(img.numpy().squeeze(), perturbed, label, pred)

        successes.append(stats["success"])
        distances.append(stats["l2_distance"])

    print(f"Attack Success Rate: {np.mean(successes)*100:.1f}%")
    print(f"Average L2 Distance: {np.mean(distances):.4f}")

    # Plot histogram
    plt.hist(distances, bins=20)
    plt.xlabel("L2 Perturbation Magnitude")
    plt.title("Attack Effectiveness")
    plt.savefig("outputs/stats.png")


if __name__ == "__main__":
    evaluate_attack()
