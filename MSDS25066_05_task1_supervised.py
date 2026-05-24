
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