
# ============================================================
# Task 5: SimCLR Pretraining
# Roll Number: MSDS25066
# ============================================================

import sys
sys.path.append("utils")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader
from dataset_splits import get_cifar10_subset, TwoViewDataset
from seed import set_seed

set_seed(2026)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================
# SimCLR Model — same as Task 4
# We copy here so this file runs independently
# ============================================================

class SimCLREncoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                                   stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        backbone.fc      = nn.Identity()
        self.encoder     = backbone

    def forward(self, x):
        return self.encoder(x)


class ProjectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        return self.projection(x)


class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder    = SimCLREncoder()
        self.projection = ProjectionHead()

    def forward(self, x):
        features    = self.encoder(x)
        projections = self.projection(features)
        return features, projections


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        N  = z1.shape[0]
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        z  = torch.cat([z1, z2], dim=0)

        sim_matrix = torch.mm(z, z.T) / self.temperature
        mask       = torch.eye(2 * N, dtype=torch.bool).to(z.device)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

        labels = torch.arange(N).to(z.device)
        labels = torch.cat([labels + N, labels], dim=0)

        return F.cross_entropy(sim_matrix, labels)

print("SimCLR classes loaded successfully")


# Augmentation pipeline — same as Task 2
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

# Load unlabeled training data
# NOTE: labels exist but we do NOT use them during SimCLR training
unlabeled_dataset = get_cifar10_subset(
    data_root  = "data",
    split_file = "splits/train_ssl_unlabeled.txt",
    train      = True,
    transform  = TwoViewTransform(simclr_transform),
    download   = False
)

train_loader = DataLoader(unlabeled_dataset, batch_size=64,
                          shuffle=True, num_workers=0)

print(f"Unlabeled samples : {len(unlabeled_dataset)}")
print(f"Batches per epoch : {len(train_loader)}")