
# ============================================================
# Task 6: Linear Probe Evaluation
# Roll Number: MSDS25066
# ============================================================

import sys
sys.path.append("utils")

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader
from dataset_splits import get_cifar10_subset
from seed import set_seed
import os

set_seed(2026)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def build_encoder():
    # Same ResNet-18 modification as before
    backbone = models.resnet18(weights=None)
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()
    backbone.fc      = nn.Identity()
    return backbone


def load_simclr_encoder():
    # Load encoder with SimCLR trained weights
    encoder = build_encoder()

    # Saved file has "encoder." prefix in keys — remove it
    state_dict = torch.load("models/simclr_encoder.pt",
                             map_location=device)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("encoder.", "")
        new_state_dict[new_key] = value

    encoder.load_state_dict(new_state_dict)

    # Freeze all weights — no updates during training
    for param in encoder.parameters():
        param.requires_grad = False

    encoder.eval()
    return encoder


def load_random_encoder():
    # Random encoder — no training, no loaded weights
    encoder = build_encoder()

    # Freeze all weights
    for param in encoder.parameters():
        param.requires_grad = False

    encoder.eval()
    return encoder


# Test both encoders
simclr_encoder = load_simclr_encoder().to(device)
random_encoder = load_random_encoder().to(device)

print("SimCLR encoder loaded and frozen")
print("Random encoder loaded and frozen")



# Transform for evaluation — no augmentation
eval_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                std =(0.2470, 0.2435, 0.2616))
])

# Load datasets — same splits as Task 1
train_dataset = get_cifar10_subset(
    data_root  = "data",
    split_file = "splits/train_labeled_10percent.txt",
    train      = True,
    transform  = eval_transform,
    download   = False
)

val_dataset = get_cifar10_subset(
    data_root  = "data",
    split_file = "splits/val.txt",
    train      = True,
    transform  = eval_transform,
    download   = False
)

test_dataset = get_cifar10_subset(
    data_root  = "data",
    split_file = "splits/test.txt",
    train      = False,
    transform  = eval_transform,
    download   = False
)

train_loader = DataLoader(train_dataset, batch_size=64,
                          shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=64,
                          shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=64,
                          shuffle=False, num_workers=0)

print(f"Train : {len(train_dataset)}")
print(f"Val   : {len(val_dataset)}")
print(f"Test  : {len(test_dataset)}")



import torch.optim as optim

def train_linear_probe(encoder, train_loader, val_loader, epochs=20):
    """
    Freeze encoder, train only a linear classifier on top.
    encoder: frozen ResNet-18 (either random or SimCLR trained)
    """
    # Linear classifier: 512 features → 10 classes
    classifier = nn.Linear(512, 10).to(device)

    # Only train classifier parameters — encoder is frozen
    optimizer = optim.Adam(classifier.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # Training
        classifier.train()
        encoder.eval()

        total_correct = 0
        total_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Extract features — no gradient for encoder
            with torch.no_grad():
                features = encoder(images)

            # Train only classifier
            logits = classifier(features)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds          = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

        train_acc = total_correct / total_samples

        # Validation
        classifier.eval()
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images   = images.to(device)
                labels   = labels.to(device)
                features = encoder(images)
                logits   = classifier(features)
                preds    = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_samples += images.size(0)

        val_acc = val_correct / val_samples

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_acc:.4f}")

    return classifier, best_val_acc

print("Linear probe function ready")




# ── Experiment A: Random Encoder ──────────────────────────
print("\nExperiment A: Random Encoder Linear Probe")
print("-" * 45)
random_classifier, _ = train_linear_probe(
    random_encoder, train_loader, val_loader, epochs=20
)

# Test accuracy
random_classifier.eval()
correct = 0
total   = 0
with torch.no_grad():
    for images, labels in test_loader:
        images   = images.to(device)
        labels   = labels.to(device)
        features = random_encoder(images)
        logits   = random_classifier(features)
        preds    = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)

random_test_acc = correct / total
print(f"\nRandom Encoder Test Accuracy: {random_test_acc:.4f}")

# ── Experiment B: SimCLR Encoder ──────────────────────────
print("\nExperiment B: SimCLR Encoder Linear Probe")
print("-" * 45)
simclr_classifier, _ = train_linear_probe(
    simclr_encoder, train_loader, val_loader, epochs=20
)

# Test accuracy
simclr_classifier.eval()
correct = 0
total   = 0
with torch.no_grad():
    for images, labels in test_loader:
        images   = images.to(device)
        labels   = labels.to(device)
        features = simclr_encoder(images)
        logits   = simclr_classifier(features)
        preds    = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)

simclr_test_acc = correct / total
print(f"\nSimCLR Encoder Test Accuracy: {simclr_test_acc:.4f}")

# ── Final Comparison ───────────────────────────────────────
print(f"\n{'='*45}")
print(f"Random Encoder Linear Probe : {random_test_acc:.4f}")
print(f"SimCLR Encoder Linear Probe : {simclr_test_acc:.4f}")
print(f"{'='*45}")






import matplotlib.pyplot as plt

# Plot comparison
models_names = ["Random\nEncoder", "SimCLR\nEncoder"]
accuracies   = [random_test_acc, simclr_test_acc]

plt.figure(figsize=(7, 5))
bars = plt.bar(models_names, accuracies, color=["gray", "steelblue"])
plt.ylabel("Test Accuracy")
plt.title("Linear Probe Evaluation")
plt.ylim(0, 1.0)

# Add accuracy numbers on top of bars
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.01,
             f"{acc:.4f}", ha="center")

plt.tight_layout()
os.makedirs("graphs", exist_ok=True)
plt.savefig("graphs/linear_probe_accuracy.png", dpi=150)
plt.close()
print("Saved: graphs/linear_probe_accuracy.png")