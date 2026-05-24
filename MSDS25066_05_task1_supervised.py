
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