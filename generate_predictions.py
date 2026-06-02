
# ============================================================
# Generate test_predictions.csv
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
import csv
import os

set_seed(2026)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load fine-tuned model — best model we have
def build_finetuned_model():
    backbone = models.resnet18(weights=None)
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()
    backbone.fc      = nn.Identity()
    model = nn.Sequential(backbone, nn.Linear(512, 10))
    model.load_state_dict(
        torch.load("models/finetuned_model.pt", map_location=device)
    )
    return model

model = build_finetuned_model().to(device)
model.eval()

# Load test dataset
eval_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                std =(0.2470, 0.2435, 0.2616))
])

test_dataset = get_cifar10_subset(
    data_root  = "data",
    split_file = "splits/test.txt",
    train      = False,
    transform  = eval_transform,
    download   = False
)

test_loader = DataLoader(test_dataset, batch_size=64,
                         shuffle=False, num_workers=0)

# Generate predictions
import torch.nn.functional as F

rows = []
image_index = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        logits = model(images)
        probs  = F.softmax(logits, dim=1)
        preds  = logits.argmax(dim=1)

        for i in range(len(labels)):
            row = [image_index,
                   labels[i].item(),
                   preds[i].item()]
            row += [round(probs[i][c].item(), 4)
                    for c in range(10)]
            rows.append(row)
            image_index += 1

# Save CSV
os.makedirs("results", exist_ok=True)
with open("results/test_predictions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "image_index", "true_label", "predicted_label",
        "prob_class_0", "prob_class_1", "prob_class_2",
        "prob_class_3", "prob_class_4", "prob_class_5",
        "prob_class_6", "prob_class_7", "prob_class_8",
        "prob_class_9"
    ])
    writer.writerows(rows)

print(f"Saved: results/test_predictions.csv")
print(f"Total predictions: {len(rows)}")