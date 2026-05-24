
# ============================================================
# Task 1: Supervised Baseline
# Roll Number: MSDS25066
# ============================================================

import sys
sys.path.append("utils")

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from dataset_splits import get_cifar10_subset
from seed import set_seed
import os

# Fix random seed
set_seed(2026)

# Device — CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Image transforms — normalize CIFAR-10 images
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                std =(0.2470, 0.2435, 0.2616))
])

# Load train subset using TA's helper function
train_dataset = get_cifar10_subset(
    data_root  = "data",
    split_file = "splits/train_labeled_10percent.txt",
    train      = True,
    transform  = transform,
    download   = False
)

print(f"Training samples: {len(train_dataset)}")

# Look at one sample
image, label = train_dataset[0]
print(f"Image shape : {image.shape}")
print(f"Label       : {label}")

# Load validation dataset
# NOTE: val.txt has indices from the TRAIN set (not test set)
val_dataset = get_cifar10_subset(
    data_root  = "data",
    split_file = "splits/val.txt",
    train      = True,       # <-- True because val comes from train set
    transform  = transform,
    download   = False
)

# Load test dataset
# NOTE: test.txt has indices from the TEST set
test_dataset = get_cifar10_subset(
    data_root  = "data",
    split_file = "splits/test.txt",
    train      = False,      # <-- False because test comes from test set
    transform  = transform,
    download   = False
)

print(f"Validation samples : {len(val_dataset)}")
print(f"Test samples       : {len(test_dataset)}")


import torchvision.models as models
import torch.nn as nn

def build_model():
    # Load ResNet-18 with NO pretrained weights
    model = models.resnet18(weights=None)

    # CIFAR-10 images are 32x32 — original ResNet was designed for 224x224
    # So we make two changes:

    # Change 1: smaller first convolution layer
    # Original: 7x7 kernel, stride 2 (too aggressive for small images)
    # Modified: 3x3 kernel, stride 1 (gentler, preserves more information)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                            stride=1, padding=1, bias=False)

    # Change 2: remove the max pooling layer
    # Original maxpool shrinks image too much for 32x32 input
    model.maxpool = nn.Identity()

    # Change 3: final layer outputs 10 classes (not 1000 like ImageNet)
    model.fc = nn.Linear(512, 10)

    return model

# Build the model
model = build_model()
print(f"\nModel built successfully")

# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")



# DataLoader wraps dataset and serves images in batches
# shuffle=True for training — randomizes order every epoch
# shuffle=False for val/test — consistent evaluation

train_loader = DataLoader(train_dataset, batch_size=64,
                          shuffle=True,  num_workers=0)

val_loader   = DataLoader(val_dataset,   batch_size=64,
                          shuffle=False, num_workers=0)

test_loader  = DataLoader(test_dataset,  batch_size=64,
                          shuffle=False, num_workers=0)

# Test the loader — get one batch and check its shape
images, labels = next(iter(train_loader))
print(f"\nOne batch of images shape : {images.shape}")
print(f"One batch of labels shape : {labels.shape}")


import torch.optim as optim

# Loss function — measures how wrong the predictions are
criterion = nn.CrossEntropyLoss()

# Optimizer — updates model weights to reduce loss
optimizer = optim.Adam(model.parameters(), lr=3e-4)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()  # training mode
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass — model makes predictions
        logits = model(images)

        # Calculate loss
        loss = criterion(logits, labels)

        # Backward pass — calculate gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track progress
        total_loss    += loss.item() * images.size(0)
        predictions    = logits.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy



def evaluate(model, loader, criterion):
    model.eval()   # evaluation mode — no dropout, no weight updates
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():   # do not calculate gradients — saves memory
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits      = model(images)
            loss        = criterion(logits, labels)

            total_loss    += loss.item() * images.size(0)
            predictions    = logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy



# Move model to device
model = model.to(device)

EPOCHS = 30

# If model already trained and saved, skip retraining to save time
# To retrain from scratch: delete supervised_best.pt and run again
if os.path.exists("models/supervised_best.pt"):
    print("\nFound saved model — skipping training")
    model.load_state_dict(torch.load("models/supervised_best.pt",
                                      map_location=device))
    # Still need these for plotting
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

else:
    # Only train if no saved model found
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    print("\nStarting training...\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader,
                                                criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/supervised_best.pt")

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f}")

    print(f"\nBest Val Accuracy: {best_val_acc:.4f}")



import matplotlib.pyplot as plt
import os

# Load the best saved model
model.load_state_dict(torch.load("models/supervised_best.pt",
                                  map_location=device))

# Evaluate on test set
test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f"\nFinal Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

# Plot loss curves only if we have training history
os.makedirs("graphs", exist_ok=True)

if len(train_losses) > 0:
    epochs_range = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs_range, train_losses, label="Train Loss")
    ax1.plot(epochs_range, val_losses,   label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Supervised Baseline - Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_range, train_accs, label="Train Accuracy")
    ax2.plot(epochs_range, val_accs,   label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Supervised Baseline - Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("graphs/supervised_loss.png", dpi=150)
    plt.close()
    print("Saved: graphs/supervised_loss.png")
else:
    print("Graph already saved from previous training run")


import sys
sys.path.append("utils")
from metrics import save_confusion_matrix

# Collect all predictions on test set
model.eval()
all_preds  = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images  = images.to(device)
        logits  = model(images)
        preds   = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

# Save confusion matrix using TA's helper
save_confusion_matrix(
    y_true   = all_labels,
    y_pred   = all_preds,
    out_path = "results/supervised_confusion_matrix.png",
    title    = "Supervised Baseline - Confusion Matrix"
)
print("Saved: results/supervised_confusion_matrix.png")