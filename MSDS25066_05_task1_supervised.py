
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

# Test with just ONE epoch to make sure it works
model = model.to(device)
train_loss, train_acc = train_one_epoch(model, train_loader,
                                        criterion, optimizer)
print(f"\nEpoch 1 — Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")




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

# Test evaluation on validation set
val_loss, val_acc = evaluate(model, val_loader, criterion)
print(f"Val   — Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")