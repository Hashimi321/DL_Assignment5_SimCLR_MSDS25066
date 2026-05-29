
# ============================================================
# Task 7: Fine-Tuning the SimCLR Encoder
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


def build_finetune_model():
    """
    Load SimCLR pretrained encoder and add
    classification head on top.
    Entire model will be trained — nothing frozen.
    """
    # Build encoder
    backbone = models.resnet18(weights=None)
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()
    backbone.fc      = nn.Identity()

    # Load SimCLR pretrained weights
    state_dict = torch.load("models/simclr_encoder.pt",
                             map_location=device)

    # Remove "encoder." prefix from keys
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("encoder.", "")
        new_state_dict[new_key] = value

    backbone.load_state_dict(new_state_dict)

    # Add classification head
    # 512 features → 10 classes
    model = nn.Sequential(
        backbone,
        nn.Linear(512, 10)
    )

    return model

# Build model
model = build_finetune_model().to(device)
print("Fine-tune model built successfully")

# Count trainable parameters
trainable = sum(p.numel() for p in model.parameters()
                if p.requires_grad)
print(f"Trainable parameters: {trainable:,}")



# Transforms
train_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomCrop(32, padding=4),
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                std =(0.2470, 0.2435, 0.2616))
])

eval_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                std =(0.2470, 0.2435, 0.2616))
])

# Load datasets
train_dataset = get_cifar10_subset(
    data_root  = "data",
    split_file = "splits/train_labeled_10percent.txt",
    train      = True,
    transform  = train_transform,
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

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

EPOCHS = 20

best_val_acc = 0.0
train_accs   = []
val_accs     = []

print("\nStarting fine-tuning...\n")

for epoch in range(1, EPOCHS + 1):

    # Training
    model.train()
    total_correct = 0
    total_samples = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds          = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    train_acc = total_correct / total_samples

    # Validation
    model.eval()
    val_correct = 0
    val_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds  = logits.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_samples += images.size(0)

    val_acc = val_correct / val_samples

    train_accs.append(train_acc)
    val_accs.append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(),
                   "models/finetuned_model.pt")

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f}")

print(f"\nBest Val Accuracy: {best_val_acc:.4f}")