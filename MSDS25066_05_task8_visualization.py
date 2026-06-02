
# ============================================================
# Task 8: PCA/t-SNE Feature Visualization
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
import numpy as np
import os

set_seed(2026)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Simple transform — no augmentation
eval_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                std =(0.2470, 0.2435, 0.2616))
])

# Load 1000 validation images
val_dataset = get_cifar10_subset(
    data_root  = "data",
    split_file = "splits/val.txt",
    train      = True,
    transform  = eval_transform,
    download   = False
)

# Take exactly 1000 images with fixed seed
from torch.utils.data import Subset
set_seed(2026)
indices     = list(range(1000))
val_1000    = Subset(val_dataset, indices)
val_loader  = DataLoader(val_1000, batch_size=64,
                         shuffle=False, num_workers=0)

print(f"Validation samples for visualization: {len(val_1000)}")




def build_encoder():
    backbone = models.resnet18(weights=None)
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()
    backbone.fc      = nn.Identity()
    return backbone


def fix_state_dict(state_dict):
    # Remove "encoder." prefix from keys if present
    new_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("encoder.", "")
        new_dict[new_key] = value
    return new_dict


# 1. Random encoder — no weights loaded
random_encoder = build_encoder().to(device)
random_encoder.eval()
print("Random encoder ready")

# 2. SimCLR encoder — load pretrained weights
simclr_encoder = build_encoder().to(device)
simclr_encoder.load_state_dict(
    fix_state_dict(torch.load("models/simclr_encoder.pt",
                               map_location=device))
)
simclr_encoder.eval()
print("SimCLR encoder ready")

# 3. Fine-tuned encoder — load from finetuned model
# Fine-tuned model is nn.Sequential(backbone, linear)
# So index 0 is the encoder
finetuned_full = nn.Sequential(
    build_encoder(),
    nn.Linear(512, 10)
)
finetuned_full.load_state_dict(
    torch.load("models/finetuned_model.pt",
                map_location=device)
)
finetuned_encoder = finetuned_full[0].to(device)
finetuned_encoder.eval()
print("Fine-tuned encoder ready")



def extract_features(encoder, loader):
    """
    Pass all images through encoder.
    Returns features and labels as numpy arrays.
    """
    all_features = []
    all_labels   = []

    with torch.no_grad():
        for images, labels in loader:
            images   = images.to(device)
            features = encoder(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_labels   = np.concatenate(all_labels,   axis=0)

    return all_features, all_labels

# Extract features from all three encoders
print("\nExtracting features...")

random_features,   random_labels   = extract_features(random_encoder,   val_loader)
simclr_features,   simclr_labels   = extract_features(simclr_encoder,   val_loader)
finetuned_features, finetuned_labels = extract_features(finetuned_encoder, val_loader)

print(f"Random features shape   : {random_features.shape}")
print(f"SimCLR features shape   : {simclr_features.shape}")
print(f"Finetuned features shape: {finetuned_features.shape}")




import sys
from visualization import save_2d_feature_plot

os.makedirs("results", exist_ok=True)

print("\nGenerating PCA visualizations...")

# 1. Random encoder PCA
save_2d_feature_plot(
    features = random_features,
    labels   = random_labels,
    out_path = "results/random_encoder_pca_or_tsne.png",
    method   = "pca",
    title    = "Random Encoder - PCA",
    seed     = 2026
)
print("Saved: results/random_encoder_pca_or_tsne.png")

# 2. SimCLR encoder PCA
save_2d_feature_plot(
    features = simclr_features,
    labels   = simclr_labels,
    out_path = "results/simclr_encoder_pca_or_tsne.png",
    method   = "pca",
    title    = "SimCLR Encoder - PCA",
    seed     = 2026
)
print("Saved: results/simclr_encoder_pca_or_tsne.png")

# 3. Fine-tuned encoder PCA
save_2d_feature_plot(
    features = finetuned_features,
    labels   = finetuned_labels,
    out_path = "results/finetuned_encoder_pca_or_tsne.png",
    method   = "pca",
    title    = "Fine-tuned Encoder - PCA",
    seed     = 2026
)
print("Saved: results/finetuned_encoder_pca_or_tsne.png")

print("\nTask 8 Complete!")