
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


import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        # Temperature controls how sharp the separation is
        # Lower = sharper separation between positive and negative
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        z1: projections from view 1 — shape [N, 128]
        z2: projections from view 2 — shape [N, 128]
        """
        N = z1.shape[0]  # batch size

        # Step 1: Normalize vectors to unit length
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Step 2: Concatenate all projections
        # Shape: [2N, 128]
        z = torch.cat([z1, z2], dim=0)

        # Step 3: Compute full similarity matrix
        # Shape: [2N, 2N]
        sim_matrix = torch.mm(z, z.T) / self.temperature

        # Step 4: Remove diagonal (similarity of vector with itself)
        mask = torch.eye(2 * N, dtype=torch.bool).to(z.device)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

        # Step 5: Positive pair labels
        # For view1[i], positive is view2[i] which is at index i+N
        labels = torch.arange(N).to(z.device)
        labels = torch.cat([labels + N, labels], dim=0)

        # Step 6: Cross entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

# Test NT-Xent loss
criterion = NTXentLoss(temperature=0.5)
dummy_z1  = torch.randn(4, 128).to(device)
dummy_z2  = torch.randn(4, 128).to(device)
loss      = criterion(dummy_z1, dummy_z2)
print(f"NT-Xent loss test: {loss.item():.4f}")
print("Expected: around 2.0 to 3.0 for random inputs")