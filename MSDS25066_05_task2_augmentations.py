
# ============================================================
# Task 2: Augmentation Visualization
# Roll Number: MSDS25066
# ============================================================

import sys
sys.path.append("utils")

import torch
import torchvision.transforms as T
from dataset_splits import get_cifar10_subset
from seed import set_seed

set_seed(2026)

# This is the exact augmentation pipeline from the assignment
simclr_transform = T.Compose([
    T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.4, contrast=0.4,
                  saturation=0.4, hue=0.1),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                std =(0.2470, 0.2435, 0.2616))
])

print("Augmentation pipeline created successfully")
print(f"Number of transforms: {len(simclr_transform.transforms)}")


# TwoViewTransform — takes one image, returns two augmented views
# This is required to be implemented by us (not from any library)
class TwoViewTransform:
    def __init__(self, transform):
        # Store the transform to apply twice
        self.transform = transform

    def __call__(self, x):
        # Apply transform twice to same image
        # Each time random operations give different result
        view1 = self.transform(x)
        view2 = self.transform(x)
        return view1, view2

# Create the two-view transform
two_view_transform = TwoViewTransform(simclr_transform)
print("TwoViewTransform created successfully")



# Simple transform for original images — no augmentation
# We just convert to tensor and normalize for display
plain_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                std =(0.2470, 0.2435, 0.2616))
])

# Load dataset with plain transform for originals
plain_dataset = get_cifar10_subset(
    data_root  = "data",
    split_file = "splits/train_labeled_10percent.txt",
    train      = True,
    transform  = plain_transform,
    download   = False
)

# Load same dataset with two-view transform for augmented views
twoview_dataset = get_cifar10_subset(
    data_root  = "data",
    split_file = "splits/train_labeled_10percent.txt",
    train      = True,
    transform  = two_view_transform,
    download   = False
)

print(f"Plain dataset size    : {len(plain_dataset)}")
print(f"Two-view dataset size : {len(twoview_dataset)}")

# Check what two-view returns
sample_views, label = twoview_dataset[0]
view1, view2 = sample_views
print(f"View 1 shape : {view1.shape}")
print(f"View 2 shape : {view2.shape}")



import sys
from visualization import save_augmentation_grid
import os

os.makedirs("results", exist_ok=True)

# Collect 10 original images and their two views
originals = []
views1    = []
views2    = []

for i in range(10):
    # Get original image
    original, _    = plain_dataset[i]
    originals.append(original)

    # Get two augmented views of same image
    (view1, view2), _ = twoview_dataset[i]
    views1.append(view1)
    views2.append(view2)

# Save the grid using TA's helper function
save_augmentation_grid(
    originals = originals,
    view1s    = views1,
    view2s    = views2,
    out_path  = "results/augmentation_examples.png",
    max_rows  = 10
)

print("Saved: results/augmentation_examples.png")