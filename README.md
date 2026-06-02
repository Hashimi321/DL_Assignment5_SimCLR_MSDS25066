# DL Assignment 5 — SimCLR
**Student:** MSDS25066  
**Course:** Deep Learning Spring 2026  
**Assignment:** Self-Supervised Learning with SimCLR on CIFAR-10

---

## Results Summary

| Model | Test Accuracy |
|-------|--------------|
| Supervised ResNet-18 (10% labels) | 53.10% |
| Random Encoder + Linear Probe | 27.81% |
| SimCLR Encoder + Linear Probe | 75.00% |
| SimCLR Encoder + Fine-Tuning | 81.61% |

---

## Feature Similarity

| Pair Type | Before SimCLR | After SimCLR |
|-----------|--------------|--------------|
| Same image (two views) | 0.9881 | 0.9245 |
| Different images | 0.9845 | 0.3221 |

---

## Training Settings

| Setting | Value |
|---------|-------|
| Dataset | CIFAR-10 |
| Encoder | ResNet-18 modified for CIFAR-10 |
| Batch size | 64 |
| SimCLR epochs | 50 |
| Linear probe epochs | 20 |
| Fine-tuning epochs | 20 |
| Learning rate | 3e-4 |
| Temperature | 0.5 |
| Optimizer | Adam |
| Random seed | 2026 |

---

## Project Structure
├── splits/          # TA provided split files
├── utils/           # TA provided helper files
├── models/          # Saved model weights
├── results/         # Output images and files
├── graphs/          # Training curves
├── data/            # CIFAR-10 dataset
└── templates/       # TA provided templates

---

## How to Run

```bash
# Task 1 - Supervised Baseline
python MSDS25066_05_task1_supervised.py

# Task 2 - Augmentations
python MSDS25066_05_task2_augmentations.py

# Task 3 - Similarity Before Training
python MSDS25066_05_task3_similarity.py

# Task 4 - SimCLR Implementation
python MSDS25066_05_task4_simclr.py

# Task 5 - SimCLR Pretraining (run on GPU)
python MSDS25066_05_task5_pretraining.py

# Task 6 - Linear Probe
python MSDS25066_05_task6_linear_probe.py

# Task 7 - Fine Tuning (run on GPU)
python MSDS25066_05_task7_finetune.py

# Task 8 - PCA Visualization
python MSDS25066_05_task8_visualization.py
```


