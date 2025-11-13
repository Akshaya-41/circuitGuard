"""
CircuitGuard - Configuration for Milestone 1
Dataset Preparation and Image Processing
"""

import os

# ============================================
# PATHS
# ============================================
BASE_DIR = r"C:\CircuitGuardd_Infosyss"

# Input
DATASET_DIR = os.path.join(BASE_DIR, "DeepPCB", "PCBData")

# Output - Milestone 1
OUTPUT_DIR = os.path.join(BASE_DIR, "milestone1_output")
ALIGNED_DIR = os.path.join(OUTPUT_DIR, "aligned_images")
DIFF_MAPS_DIR = os.path.join(OUTPUT_DIR, "difference_maps")
MASKS_DIR = os.path.join(OUTPUT_DIR, "defect_masks")
HIGHLIGHTED_DIR = os.path.join(OUTPUT_DIR, "highlighted_defects")
CONTOURS_DIR = os.path.join(OUTPUT_DIR, "contour_visualizations")
ROIS_DIR = os.path.join(OUTPUT_DIR, "extracted_rois")
SAMPLES_DIR = os.path.join(OUTPUT_DIR, "sample_outputs")

# Create all directories
for dir_path in [OUTPUT_DIR, ALIGNED_DIR, DIFF_MAPS_DIR, MASKS_DIR, 
                 HIGHLIGHTED_DIR, CONTOURS_DIR, ROIS_DIR, SAMPLES_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ============================================
# DEFECT TYPE MAPPING (for labeling)
# ============================================
DEFECT_TYPES = {
    '1': 'open',
    '2': 'short',
    '3': 'mousebite',
    '4': 'spur',
    '5': 'spurious_copper',
    '6': 'pin_hole'
}

# ============================================
# IMAGE PROCESSING PARAMETERS
# ============================================
# Alignment
MAX_FEATURES = 1000
MATCH_RATIO = 0.7

# Defect detection
GAUSSIAN_KERNEL = (5, 5)
MIN_DEFECT_AREA = 30
MAX_DEFECT_AREA = 10000

# Morphological operations
MORPH_KERNEL_OPEN = (3, 3)
MORPH_KERNEL_CLOSE = (7, 7)
MORPH_ITERATIONS_OPEN = 2
MORPH_ITERATIONS_CLOSE = 2

# ROI extraction
PADDING = 15  # Padding around bounding boxes
IOU_THRESHOLD = 0.3  # For matching to ground truth
ROI_RESIZE = 128  # Standard size for extracted ROIs

# ============================================
# VISUALIZATION
# ============================================
NUM_SAMPLES = 6  # Number of samples to visualize
HIGHLIGHT_COLOR = (0, 0, 255)  # Red for defects (BGR format)
CONTOUR_COLOR = (0, 255, 0)  # Green for contours (BGR format)
CONTOUR_THICKNESS = 2

# ============================================
# MISC
# ============================================
SEED = 42

print("="*70)
print(" CIRCUITGUARD - MILESTONE 1 CONFIGURATION")
print("="*70)
print(f"📁 Dataset Directory: {DATASET_DIR}")
print(f"📁 Output Directory: {OUTPUT_DIR}")
print(f"🎯 Target: Complete Dataset Preparation & Image Processing")
print("="*70)