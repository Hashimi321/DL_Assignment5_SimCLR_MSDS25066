
# ============================================================
# Task 4: SimCLR Implementation
# Roll Number: MSDS25066
# ============================================================

import sys
sys.path.append("utils")

import torch
import torch.nn as nn
import torchvision.models as models
from seed import set_seed

set_seed(2026)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



class SimCLREncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Load ResNet-18 with no pretrained weights
        backbone = models.resnet18(weights=None)

        # Same CIFAR-10 modifications as before
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                                   stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()

        # Remove final classification layer
        # We want 512 features as output
        backbone.fc = nn.Identity()

        self.encoder = backbone

    def forward(self, x):
        # Input : [batch, 3, 32, 32]
        # Output: [batch, 512]
        return self.encoder(x)

# Test encoder
encoder = SimCLREncoder().to(device)
dummy = torch.randn(4, 3, 32, 32).to(device)
out   = encoder(dummy)
print(f"Encoder input  shape: {dummy.shape}")
print(f"Encoder output shape: {out.shape}")


class ProjectionHead(nn.Module):
    def __init__(self):
        super().__init__()

        # Assignment specification:
        # Linear(512 -> 256), ReLU, Linear(256 -> 128)
        self.projection = nn.Sequential(
            nn.Linear(512, 256),  # first linear layer
            nn.ReLU(),            # activation function
            nn.Linear(256, 128)   # second linear layer
        )

    def forward(self, x):
        # Input : [batch, 512]
        # Output: [batch, 128]
        return self.projection(x)

# Test projection head
proj_head = ProjectionHead().to(device)
dummy_features = torch.randn(4, 512).to(device)
proj_out       = proj_head(dummy_features)
print(f"Projection input  shape: {dummy_features.shape}")
print(f"Projection output shape: {proj_out.shape}")


class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder    = SimCLREncoder()
        self.projection = ProjectionHead()

    def forward(self, x):
        # Step 1: Extract features from image
        # Input : [batch, 3, 32, 32]
        # Output: [batch, 512]
        features = self.encoder(x)

        # Step 2: Project to smaller space
        # Input : [batch, 512]
        # Output: [batch, 128]
        projections = self.projection(features)

        return features, projections

# Test full SimCLR model
simclr_model = SimCLR().to(device)
dummy_images          = torch.randn(4, 3, 32, 32).to(device)
features, projections = simclr_model(dummy_images)
print(f"SimCLR input       shape: {dummy_images.shape}")
print(f"SimCLR features    shape: {features.shape}")
print(f"SimCLR projections shape: {projections.shape}")