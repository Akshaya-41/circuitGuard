"""
CircuitGuard - Milestone 2 Utilities
Helper functions for training and evaluation
"""

import os
import random
import numpy as np
import torch
import json
from milestone2_config import SEED

# ============================================
# REPRODUCIBILITY
# ============================================
def set_seed(seed=SEED):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed()

# ============================================
# FILE I/O
# ============================================
def save_json(data, filepath):
    """Save data to JSON file"""
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return obj
    
    # Recursively convert all values
    def recursive_convert(data):
        if isinstance(data, dict):
            return {k: recursive_convert(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [recursive_convert(item) for item in data]
        else:
            return convert(data)
    
    converted_data = recursive_convert(data)
    
    with open(filepath, 'w') as f:
        json.dump(converted_data, f, indent=4)


def load_json(filepath):
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


# ============================================
# TRAINING UTILITIES
# ============================================
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_accuracy(outputs, targets):
    """Calculate accuracy"""
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return 100.0 * correct / total


# ============================================
# MODEL UTILITIES
# ============================================
def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_checkpoint(state, filepath):
    """Save model checkpoint"""
    torch.save(state, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


# ============================================
# EVALUATION UTILITIES
# ============================================
def compute_class_accuracy(confusion_matrix):
    """Compute per-class accuracy from confusion matrix"""
    class_acc = []
    for i in range(len(confusion_matrix)):
        if confusion_matrix[i].sum() > 0:
            acc = 100.0 * confusion_matrix[i, i] / confusion_matrix[i].sum()
            class_acc.append(acc)
        else:
            class_acc.append(0.0)
    return class_acc


def format_time(seconds):
    """Format seconds to readable time string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


# ============================================
# DATA UTILITIES
# ============================================
def get_class_distribution(dataset):
    """Get distribution of classes in dataset"""
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    else:
        targets = [label for _, label in dataset.samples]
    
    unique, counts = np.unique(targets, return_counts=True)
    return dict(zip(unique, counts))


def print_dataset_info(train_dataset, val_dataset, test_dataset):
    """Print detailed dataset information"""
    print("\n" + "="*70)
    print(" DATASET INFORMATION")
    print("="*70)
    
    print(f"\n📊 Dataset Sizes:")
    print(f"  Train: {len(train_dataset):,} images")
    print(f"  Val:   {len(val_dataset):,} images")
    print(f"  Test:  {len(test_dataset):,} images")
    print(f"  Total: {len(train_dataset) + len(val_dataset) + len(test_dataset):,} images")
    
    print(f"\n🏷️  Classes ({len(train_dataset.classes)}):")
    for idx, class_name in enumerate(train_dataset.classes):
        print(f"  {idx}: {class_name}")
    
    print(f"\n📈 Class Distribution:")
    
    train_dist = get_class_distribution(train_dataset)
    val_dist = get_class_distribution(val_dataset)
    test_dist = get_class_distribution(test_dataset)
    
    print(f"\n  {'Class':<20} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    
    for idx, class_name in enumerate(train_dataset.classes):
        train_count = train_dist.get(idx, 0)
        val_count = val_dist.get(idx, 0)
        test_count = test_dist.get(idx, 0)
        total_count = train_count + val_count + test_count
        
        print(f"  {class_name:<20} {train_count:>8} {val_count:>8} {test_count:>8} {total_count:>8}")
    
    print("="*70)


print("✅ Milestone 2 utilities loaded!")