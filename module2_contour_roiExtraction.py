"""
CircuitGuard - Module 2: Contour Detection and ROI Extraction
Tasks:
- Use OpenCV to detect contours of defects
- Extract bounding boxes and crop individual defect regions
- Label defect ROIs for model training

Deliverables:
- ROI extraction pipeline
- Cropped and labeled defect samples
- Visualization of defect contours
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt

from config import *
from utils import (load_json, save_json, load_annotations,
                   detect_contours_and_boxes, draw_contours_and_boxes,
                   extract_rois, match_roi_to_annotation)

print("="*70)
print(" MODULE 2: CONTOUR DETECTION AND ROI EXTRACTION")
print("="*70)

def module2_complete_pipeline():
    """
    Complete Module 2: Contour detection, bounding boxes, and ROI extraction
    """
    
    # ============================================
    # LOAD MODULE 1 RESULTS
    # ============================================
    print("\n[Preparation] Loading Module 1 results...")
    
    module1_path = os.path.join(OUTPUT_DIR, 'module1_results.json')
    if not os.path.exists(module1_path):
        print("❌ Module 1 results not found!")
        print("   Please run module1_image_subtraction.py first")
        return None
    
    module1_data = load_json(module1_path)
    detections = module1_data['results']
    print(f"✓ Loaded {len(detections)} processed images from Module 1")
    
    # ============================================
    # TASK 1: DETECT CONTOURS
    # ============================================
    print("\n[Task 1/3] Detecting contours of defects using OpenCV...")
    
    contour_results = []
    total_contours = 0
    
    for detection in tqdm(detections, desc="Detecting contours"):
        try:
            # Load mask
            mask = cv2.imread(detection['mask'], cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                continue
            
            # Detect contours and bounding boxes
            boxes = detect_contours_and_boxes(
                mask, 
                min_area=MIN_DEFECT_AREA,
                max_area=MAX_DEFECT_AREA
            )
            
            total_contours += len(boxes)
            
            contour_results.append({
                'name': detection['name'],
                'group': detection['group'],
                'aligned': detection['aligned'],
                'mask': detection['mask'],
                'annotation': detection['annotation'],
                'boxes': boxes,
                'num_contours': len(boxes)
            })
            
        except Exception as e:
            print(f"\n❌ Error: {detection['name']}: {e}")
    
    print(f"✓ Detected {total_contours} contours across {len(contour_results)} images")
    print(f"✓ Average contours per image: {total_contours/len(contour_results):.2f}")
    
    # ============================================
    # TASK 2: EXTRACT BOUNDING BOXES & CROP ROIs
    # ============================================
    print("\n[Task 2/3] Extracting bounding boxes and cropping defect regions...")
    
    all_rois = []
    roi_counter = Counter()
    matched_count = 0
    unmatched_count = 0
    
    for result in tqdm(contour_results, desc="Extracting ROIs"):
        try:
            # Load aligned image
            aligned = cv2.imread(result['aligned'])
            
            if aligned is None:
                continue
            
            # Load ground truth annotations
            annotations = load_annotations(result['annotation'])
            
            # Extract ROIs from bounding boxes
            rois = extract_rois(aligned, result['boxes'], padding=PADDING)
            
            # Match each ROI to ground truth and label
            for roi in rois:
                # Match to annotation
                defect_type_id, iou = match_roi_to_annotation(
                    roi['bbox'], 
                    annotations,
                    IOU_THRESHOLD
                )
                
                if defect_type_id and defect_type_id in DEFECT_TYPES:
                    defect_name = DEFECT_TYPES[defect_type_id]
                    
                    # Resize ROI to standard size
                    roi_resized = cv2.resize(roi['image'], (ROI_RESIZE, ROI_RESIZE))
                    
                    roi_data = {
                        'image': roi_resized,
                        'defect_type': defect_name,
                        'type_id': defect_type_id,
                        'bbox': roi['bbox'],
                        'width': roi['width'],
                        'height': roi['height'],
                        'area': roi['area'],
                        'iou': iou,
                        'source_image': result['name'],
                        'contour_idx': roi['contour_idx']
                    }
                    
                    all_rois.append(roi_data)
                    roi_counter[defect_name] += 1
                    matched_count += 1
                else:
                    unmatched_count += 1
            
            # Create visualization with contours and boxes
            visualize_contours_boxes(result, aligned)
            
        except Exception as e:
            print(f"\n❌ Error: {result['name']}: {e}")
    
    print(f"\n✓ Extracted {len(all_rois)} labeled ROIs")
    print(f"  Matched to ground truth: {matched_count}")
    print(f"  Unmatched: {unmatched_count}")
    print(f"  Match rate: {100*matched_count/(matched_count+unmatched_count):.1f}%")
    
    # ============================================
    # TASK 3: LABEL AND SAVE ROIs
    # ============================================
    print("\n[Task 3/3] Labeling and saving defect ROIs...")
    
    # Create class folders
    for defect_name in DEFECT_TYPES.values():
        os.makedirs(os.path.join(ROIS_DIR, defect_name), exist_ok=True)
    
    # Save ROIs
    saved_rois = []
    
    for idx, roi_data in enumerate(tqdm(all_rois, desc="Saving ROIs")):
        defect_type = roi_data['defect_type']
        filename = f"roi_{idx:06d}.jpg"
        
        # Save ROI image
        save_path = os.path.join(ROIS_DIR, defect_type, filename)
        cv2.imwrite(save_path, roi_data['image'])
        
        # Store metadata
        saved_rois.append({
            'filename': filename,
            'defect_type': defect_type,
            'type_id': roi_data['type_id'],
            'bbox': roi_data['bbox'],
            'width': roi_data['width'],
            'height': roi_data['height'],
            'area': roi_data['area'],
            'iou': roi_data['iou'],
            'source_image': roi_data['source_image']
        })
    
    print(f"✓ Saved {len(saved_rois)} ROI images")
    
    # Print distribution
    print("\n📊 ROI Distribution by Defect Type:")
    for defect_name in sorted(DEFECT_TYPES.values()):
        count = roi_counter[defect_name]
        percentage = 100 * count / len(all_rois) if all_rois else 0
        print(f"  {defect_name:20s}: {count:5d} ({percentage:5.1f}%)")
    
    # ============================================
    # SAVE METADATA
    # ============================================
    print("\n[4/4] Saving module metadata...")
    
    metadata = {
        'module': 'Module 2: Contour Detection and ROI Extraction',
        'total_images_processed': len(contour_results),
        'total_contours_detected': total_contours,
        'total_rois_extracted': len(all_rois),
        'matched_rois': matched_count,
        'unmatched_rois': unmatched_count,
        'match_rate': matched_count / (matched_count + unmatched_count) if (matched_count + unmatched_count) > 0 else 0,
        'roi_distribution': dict(roi_counter),
        'rois': saved_rois
    }
    
    save_json(metadata, os.path.join(OUTPUT_DIR, 'module2_results.json'))
    print(f"✓ Metadata saved: {OUTPUT_DIR}/module2_results.json")
    
    # ============================================
    # CREATE SAMPLE VISUALIZATIONS
    # ============================================
    print("\n[5/5] Creating sample visualizations...")
    create_module2_samples(all_rois[:24])  # Show 24 samples
    
    # ============================================
    # PRINT SUMMARY
    # ============================================
    print("\n" + "="*70)
    print(" MODULE 2 DELIVERABLES - COMPLETE!")
    print("="*70)
    
    print("\n✅ Deliverable 1: ROI extraction pipeline")
    print(f"   📝 This script (module2_contour_roi_extraction.py)")
    print(f"   📊 Total ROIs extracted: {len(all_rois)}")
    
    print("\n✅ Deliverable 2: Cropped and labeled defect samples")
    print(f"   📁 Location: {ROIS_DIR}")
    print(f"   📂 Organized by defect type:")
    for defect_name in sorted(DEFECT_TYPES.values()):
        print(f"      ├── {defect_name}/ ({roi_counter[defect_name]} images)")
    
    print("\n✅ Deliverable 3: Visualization of defect contours")
    print(f"   📁 Contour visualizations: {CONTOURS_DIR}")
    print(f"   📁 Sample ROIs: {SAMPLES_DIR}")
    
    print("\n📊 Module 2 Evaluation Metrics:")
    print(f"   ✓ ROI detection precision: {100*matched_count/(matched_count+unmatched_count):.1f}%")
    print(f"   ✓ Bounding box accuracy: Verified via IoU matching (threshold={IOU_THRESHOLD})")
    print(f"   ✓ Total contours detected: {total_contours}")
    print(f"   ✓ Successfully labeled ROIs: {matched_count}")
    
    print("\n" + "="*70)
    
    return metadata


def visualize_contours_boxes(result, image):
    """Create visualization showing contours and bounding boxes"""
    
    # Draw contours and boxes
    visualization = draw_contours_and_boxes(
        image, 
        result['boxes'],
        contour_color=CONTOUR_COLOR,
        box_color=(255, 0, 0),  # Red boxes
        thickness=CONTOUR_THICKNESS
    )
    
    # Save visualization
    save_path = os.path.join(CONTOURS_DIR, f"{result['name']}_contours.jpg")
    cv2.imwrite(save_path, visualization)


def create_module2_samples(sample_rois):
    """Create grid visualization of extracted ROIs"""
    
    if len(sample_rois) == 0:
        return
    
    # Create grid (4 rows x 6 columns)
    num_samples = min(24, len(sample_rois))
    rows = 4
    cols = 6
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx in range(num_samples):
        roi = sample_rois[idx]
        img_rgb = cv2.cvtColor(roi['image'], cv2.COLOR_BGR2RGB)
        
        axes[idx].imshow(img_rgb)
        axes[idx].set_title(f"{roi['defect_type']}\n{roi['width']}x{roi['height']}px", 
                           fontsize=8, fontweight='bold')
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(num_samples, rows * cols):
        axes[idx].axis('off')
    
    plt.suptitle('Module 2: Extracted and Labeled ROI Samples', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SAMPLES_DIR, 'module2_roi_samples.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ ROI samples saved: {SAMPLES_DIR}/module2_roi_samples.png")
    
    # Also create per-class visualization
    create_per_class_visualization(sample_rois)


def create_per_class_visualization(all_rois):
    """Create visualization showing samples from each defect class"""
    
    # Group ROIs by class
    rois_by_class = {}
    for roi in all_rois:
        defect_type = roi['defect_type']
        if defect_type not in rois_by_class:
            rois_by_class[defect_type] = []
        rois_by_class[defect_type].append(roi)
    
    # Create figure
    num_classes = len(rois_by_class)
    samples_per_class = 6
    
    fig, axes = plt.subplots(num_classes, samples_per_class, 
                            figsize=(18, 3*num_classes))
    
    if num_classes == 1:
        axes = axes.reshape(1, -1)
    
    for class_idx, (defect_type, rois) in enumerate(sorted(rois_by_class.items())):
        for sample_idx in range(samples_per_class):
            if sample_idx < len(rois):
                roi = rois[sample_idx]
                img_rgb = cv2.cvtColor(roi['image'], cv2.COLOR_BGR2RGB)
                axes[class_idx, sample_idx].imshow(img_rgb)
                axes[class_idx, sample_idx].axis('off')
            else:
                axes[class_idx, sample_idx].axis('off')
        
        # Add class label on the left
        axes[class_idx, 0].set_ylabel(defect_type, fontsize=12, fontweight='bold', rotation=0, 
                                      labelpad=60, va='center')
    
    plt.suptitle('ROI Samples by Defect Class', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SAMPLES_DIR, 'module2_class_samples.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Class samples saved: {SAMPLES_DIR}/module2_class_samples.png")


if __name__ == "__main__":
    print("\nStarting Module 2 Pipeline...")
    
    metadata = module2_complete_pipeline()
    
    if metadata:
        print("\n✅ MODULE 2 COMPLETE!")
        print("\n🎉 MILESTONE 1 COMPLETE!")
        print("\n📂 Final Output Structure:")
        print(f"   {OUTPUT_DIR}/")
        print(f"   ├── aligned_images/          (Module 1)")
        print(f"   ├── difference_maps/         (Module 1)")
        print(f"   ├── defect_masks/            (Module 1)")
        print(f"   ├── highlighted_defects/     (Module 1)")
        print(f"   ├── contour_visualizations/  (Module 2)")
        print(f"   ├── extracted_rois/          (Module 2)")
        print(f"   │   ├── open/")
        print(f"   │   ├── short/")
        print(f"   │   ├── mousebite/")
        print(f"   │   ├── spur/")
        print(f"   │   ├── spurious_copper/")
        print(f"   │   └── pin_hole/")
        print(f"   ├── sample_outputs/")
        print(f"   ├── module1_results.json")
        print(f"   └── module2_results.json")
        print("\n🎯 Ready for Milestone 2: Model Training and Evaluation")
    else:
        print("\n❌ Module 2 failed.")