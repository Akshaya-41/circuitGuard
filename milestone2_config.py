"""
CircuitGuard - Milestone 2 Configuration
Model Training and Evaluation Settings
"""

import os
import torch

# ============================================
# PATHS
# ============================================
BASE_DIR = r"C:\CircuitGuardd_Infosyss"

# Input: ROIs from Milestone 1
ROIS_DIR = os.path.join(BASE_DIR, "milestone1_output", "extracted_rois")

# Output: Milestone 2
OUTPUT_DIR = os.path.join(BASE_DIR, "milestone2_output")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")

# Create directories
for dir_path in [OUTPUT_DIR, MODELS_DIR, PLOTS_DIR, RESULTS_DIR, PREDICTIONS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Model save path
MODEL_PATH = os.path.join(MODELS_DIR, "mobilenetv2_best.pth")

# ============================================
# DATASET SPLIT
# ============================================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============================================
# MODEL CONFIGURATION
# ============================================
MODEL_NAME = "MobileNetV2"
IMAGE_SIZE = 128
NUM_CLASSES = 6  # open, short, mousebite, spur, spurious_copper, pin_hole

# ============================================
# TRAINING HYPERPARAMETERS
# ============================================
BATCH_SIZE = 32  # Good balance for speed and memory
EPOCHS = 30
LEARNING_RATE = 0.001  # Higher LR for faster convergence
WEIGHT_DECAY = 1e-4

# Learning rate scheduler
LR_SCHEDULER = "ReduceLROnPlateau"
LR_PATIENCE = 3
LR_FACTOR = 0.5
MIN_LR = 1e-6

# Early stopping
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 7

# ============================================
# DATA AUGMENTATION
# ============================================
# Training augmentation
AUG_HORIZONTAL_FLIP = 0.5
AUG_VERTICAL_FLIP = 0.5
AUG_ROTATION = 20
AUG_BRIGHTNESS = 0.2
AUG_CONTRAST = 0.2
AUG_SATURATION = 0.2

# ============================================
# HARDWARE SETTINGS
# ============================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0  # Set to 0 for Windows, 4 for Linux/Mac
PIN_MEMORY = True if torch.cuda.is_available() else False

# ============================================
# REPRODUCIBILITY
# ============================================
SEED = 42

# ============================================
# EVALUATION
# ============================================
TARGET_ACCURACY = 97.0  # Target: ≥97% accuracy
CONFIDENCE_THRESHOLD = 0.5

# ============================================
# DEFECT TYPES (must match Milestone 1)
# ============================================
DEFECT_TYPES = {
    '1': 'open',
    '2': 'short',
    '3': 'mousebite',
    '4': 'spur',
    '5': 'spurious_copper',
    '6': 'pin_hole'
}

DEFECT_NAMES = sorted(DEFECT_TYPES.values())  # For consistent ordering

print("="*70)
print(" CIRCUITGUARD - MILESTONE 2 CONFIGURATION")
print("="*70)
print(f"🎯 Target: ≥{TARGET_ACCURACY}% Test Accuracy")
print(f"⚡ Model: {MODEL_NAME}")
print(f"🖥️  Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
print(f"📁 ROIs Directory: {ROIS_DIR}")
print(f"📁 Output Directory: {OUTPUT_DIR}")
print(f"⏱️  Expected Training Time: <1 hour (on GPU), 2-3 hours (on CPU)")
print("="*70)