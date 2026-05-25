
# ============================================================
# Task 3: Feature Similarity Before Training
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

set_seed(2026)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



def build_encoder():
    # Load ResNet-18 with NO pretrained weights
    model = models.resnet18(weights=None)

    # Same CIFAR-10 modifications as Task 1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                            stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    # Remove final classification layer
    # We want 512 features only — not class predictions
    model.fc = nn.Identity()

    return model

# Build random encoder — no training, random weights
encoder = build_encoder().to(device)
encoder.eval()

print("Random encoder built successfully")
print("This encoder has never been trained — weights are random")


# Two-view transform — same as Task 2
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

class TwoViewTransform:
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, x):
        return self.transform(x), self.transform(x)

# Load small batch of images with two views
dataset = get_cifar10_subset(
    data_root  = "data",
    split_file = "splits/train_labeled_10percent.txt",
    train      = True,
    transform  = TwoViewTransform(simclr_transform),
    download   = False
)

loader = DataLoader(dataset, batch_size=64,
                    shuffle=False, num_workers=0)

# Get one batch
(view1, view2), labels = next(iter(loader))
view1 = view1.to(device)
view2 = view2.to(device)

print(f"View 1 batch shape : {view1.shape}")
print(f"View 2 batch shape : {view2.shape}")

# Extract features using random encoder
with torch.no_grad():
    features1 = encoder(view1)  # shape: [64, 512]
    features2 = encoder(view2)  # shape: [64, 512]

print(f"Features 1 shape : {features1.shape}")
print(f"Features 2 shape : {features2.shape}")


import torch.nn.functional as F

def cosine_similarity_pairs(f1, f2):
    """
    Compute cosine similarity between matching pairs.
    f1 shape: [N, 512]
    f2 shape: [N, 512]
    Returns average similarity score between all pairs.
    """
    # Normalize each vector to unit length
    # After this, dot product = cosine similarity
    f1_norm = F.normalize(f1, dim=1)
    f2_norm = F.normalize(f2, dim=1)

    # Multiply matching pairs and sum → cosine similarity per pair
    # shape: [N]
    similarities = (f1_norm * f2_norm).sum(dim=1)

    return similarities.mean().item()

# Similarity between two views of SAME image
same_sim = cosine_similarity_pairs(features1, features2)
print(f"Same image  (two views) similarity : {same_sim:.4f}")

# Similarity between DIFFERENT images
# Compare view1 of image i with view1 of image i+1
different_sim = cosine_similarity_pairs(features1[:-1], features1[1:])
print(f"Different images similarity        : {different_sim:.4f}")

import matplotlib.pyplot as plt
import os

def compute_similarity_matrix(f1, f2):
    """
    Compute full 2N x 2N cosine similarity matrix.
    Concatenate view1 and view2 features, then compute all pairs.
    """
    # Combine both views → shape: [2N, 512]
    all_features = torch.cat([f1, f2], dim=0)

    # Normalize
    all_features = F.normalize(all_features, dim=1)

    # Matrix multiply → shape: [2N, 2N]
    sim_matrix = torch.mm(all_features, all_features.T)

    return sim_matrix.detach().cpu().numpy()

# Use small batch of 8 images for clear visualization
small_view1 = view1[:8]
small_view2 = view2[:8]

with torch.no_grad():
    small_f1 = encoder(small_view1)
    small_f2 = encoder(small_view2)

sim_matrix = compute_similarity_matrix(small_f1, small_f2)

# Plot heatmap
os.makedirs("results", exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(sim_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im)
ax.set_title("Similarity Matrix Before SimCLR Training")
ax.set_xlabel("Image Index")
ax.set_ylabel("Image Index")
plt.tight_layout()
plt.savefig("results/similarity_matrix_before_training.png", dpi=150)
plt.close()
print("Saved: results/similarity_matrix_before_training.png")