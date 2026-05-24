
# test_setup.py
# Purpose: Check that all TA files and packages work correctly

import sys
sys.path.append("utils")  # tells Python to look in utils folder

from seed import set_seed
from dataset_splits import read_split_indices

# Set the seed
set_seed(2026)
print("Seed set successfully")

# Load one split file and print how many indices it has
train_indices = read_split_indices("splits/train_labeled_10percent.txt")
print(f"Labeled training samples: {len(train_indices)}")

val_indices = read_split_indices("splits/val.txt")
print(f"Validation samples: {len(val_indices)}")

test_indices = read_split_indices("splits/test.txt")
print(f"Test samples: {len(test_indices)}")

print("\nSetup is working correctly!")