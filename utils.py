"""
CircuitGuard - Utility Functions for Milestone 1
"""

import os
import cv2
import numpy as np
import json

# ============================================
# FILE I/O
# ============================================
def load_annotations(annotation_file):
    """
    Parse annotation file with format: x1 y1 x2 y2 defect_type_id
    Returns list of defect dictionaries
    """
    defects = []
    if not os.path.exists(annotation_file):
        return defects
    
    try:
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    x1, y1, x2, y2 = map(int, parts[0:4])
                    defect_type_id = parts[4]
                    
                    w = x2 - x1
                    h = y2 - y1
                    
                    if w > 0 and h > 0:
                        defects.append({
                            'type_id': defect_type_id,
                            'bbox': (x1, y1, x2, y2),
                            'x1': x1, 'y1': y1,
                            'x2': x2, 'y2': y2,
                            'width': w, 'height': h,
                            'area': w * h
                        })
    except Exception as e:
        print(f"Error reading {annotation_file}: {e}")
    
    return defects


def save_json(data, filepath):
    """Save data to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filepath):
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_image_pairs(base_dir):
    """
    Scan dataset directory and return list of (template, test, annotation) pairs
    """
    pairs = []
    
    for group in os.listdir(base_dir):
        group_path = os.path.join(base_dir, group)
        if not os.path.isdir(group_path):
            continue
        
        group_num = group.replace("group", "")
        images_folder = os.path.join(group_path, group_num)
        annotations_folder = os.path.join(group_path, f"{group_num}_not")
        
        if not os.path.exists(images_folder):
            continue
        
        all_files = os.listdir(images_folder)
        temp_files = [f for f in all_files if f.endswith('_temp.jpg')]
        
        for temp_file in temp_files:
            test_file = temp_file.replace('_temp.jpg', '_test.jpg')
            
            if test_file not in all_files:
                continue
            
            base_name = temp_file.replace('_temp.jpg', '')
            
            pair = {
                'template': os.path.join(images_folder, temp_file),
                'test': os.path.join(images_folder, test_file),
                'annotation': os.path.join(annotations_folder, f"{base_name}.txt"),
                'name': f"{group}_{base_name}",
                'group': group
            }
            
            pairs.append(pair)
    
    return pairs


# ============================================
# IMAGE ALIGNMENT
# ============================================
def align_images_orb(template, test_img, max_features=1000):
    """
    Align test image to template using ORB feature matching
    Returns: aligned_image, success_flag
    """
    if template.shape != test_img.shape:
        test_img = cv2.resize(test_img, (template.shape[1], template.shape[0]))
    
    # Convert to grayscale
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    template_gray = clahe.apply(template_gray)
    test_gray = clahe.apply(test_gray)
    
    # ORB feature detection
    orb = cv2.ORB_create(max_features)
    kp1, desc1 = orb.detectAndCompute(template_gray, None)
    kp2, desc2 = orb.detectAndCompute(test_gray, None)
    
    if desc1 is None or desc2 is None or len(kp1) < 10 or len(kp2) < 10:
        return test_img, False
    
    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)
    
    # Apply ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < 10:
        return test_img, False
    
    # Extract matched points
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Find homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        return test_img, False
    
    # Warp image
    h, w = template.shape[:2]
    aligned = cv2.warpPerspective(test_img, H, (w, h))
    
    return aligned, True


# ============================================
# IMAGE SUBTRACTION & THRESHOLDING
# ============================================
def compute_difference_map(template, test_img):
    """Compute absolute difference between aligned images"""
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(template_gray, test_gray)
    return diff


def apply_otsu_threshold(diff_map, kernel_size=(5, 5)):
    """Apply Gaussian blur + Otsu's thresholding"""
    blurred = cv2.GaussianBlur(diff_map, kernel_size, 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def refine_mask_morphology(binary_mask, kernel_open=(3, 3), kernel_close=(7, 7), 
                           iter_open=2, iter_close=2, min_area=30):
    """Apply morphological operations to clean up mask"""
    # Opening: remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_open)
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=iter_open)
    
    # Closing: fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_close)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iter_close)
    
    # Remove small components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    
    refined = np.zeros_like(closed)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            refined[labels == i] = 255
    
    return refined


# ============================================
# DEFECT HIGHLIGHTING
# ============================================
def highlight_defects(test_img, defect_mask, color=(0, 0, 255)):
    """Overlay defect mask on original image with colored regions"""
    overlay = test_img.copy()
    
    # Find contours
    contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create colored mask
    mask_colored = np.zeros_like(test_img)
    cv2.drawContours(mask_colored, contours, -1, color, -1)
    
    # Blend
    highlighted = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
    
    # Draw contours
    cv2.drawContours(highlighted, contours, -1, color, 2)
    
    return highlighted, len(contours)


# ============================================
# CONTOUR DETECTION & BOUNDING BOXES
# ============================================
def detect_contours_and_boxes(mask, min_area=30, max_area=10000):
    """
    Detect contours and extract bounding boxes
    Returns list of bounding boxes with metadata
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        if area < min_area or area > max_area:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        boxes.append({
            'contour': contour,
            'bbox': (x, y, w, h),
            'x1': x, 'y1': y,
            'x2': x + w, 'y2': y + h,
            'area': area,
            'contour_idx': idx
        })
    
    return boxes


def draw_contours_and_boxes(image, boxes, contour_color=(0, 255, 0), 
                            box_color=(255, 0, 0), thickness=2):
    """Draw contours and bounding boxes on image"""
    result = image.copy()
    
    for box in boxes:
        # Draw contour
        cv2.drawContours(result, [box['contour']], -1, contour_color, thickness)
        
        # Draw bounding box
        cv2.rectangle(result, (box['x1'], box['y1']), (box['x2'], box['y2']), 
                     box_color, thickness)
        
        # Add label
        label = f"#{box['contour_idx']+1}"
        cv2.putText(result, label, (box['x1'], box['y1']-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
    
    return result


# ============================================
# ROI EXTRACTION
# ============================================
def extract_rois(image, boxes, padding=15):
    """
    Extract ROI images from bounding boxes with padding
    """
    h, w = image.shape[:2]
    rois = []
    
    for box in boxes:
        x1 = max(0, box['x1'] - padding)
        y1 = max(0, box['y1'] - padding)
        x2 = min(w, box['x2'] + padding)
        y2 = min(h, box['y2'] + padding)
        
        roi_img = image[y1:y2, x1:x2]
        
        if roi_img.size == 0:
            continue
        
        rois.append({
            'image': roi_img,
            'bbox': (x1, y1, x2, y2),
            'width': x2 - x1,
            'height': y2 - y1,
            'area': box['area'],
            'contour_idx': box['contour_idx']
        })
    
    return rois


# ============================================
# BBOX MATCHING (for labeling)
# ============================================
def compute_iou(box1, box2):
    """
    Compute IoU between two boxes
    box format: (x1, y1, x2, y2)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Intersection
    ix1 = max(x1_1, x1_2)
    iy1 = max(y1_1, y1_2)
    ix2 = min(x2_1, x2_2)
    iy2 = min(y2_1, y2_2)
    
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    
    intersection = (ix2 - ix1) * (iy2 - iy1)
    
    # Union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def match_roi_to_annotation(roi_bbox, annotations, iou_threshold=0.3):
    """
    Match detected ROI to ground truth annotation
    Returns: (defect_type_id, best_iou)
    """
    best_match = None
    best_iou = 0
    
    for ann in annotations:
        ann_box = (ann['x1'], ann['y1'], ann['x2'], ann['y2'])
        iou = compute_iou(roi_bbox, ann_box)
        
        if iou > best_iou and iou >= iou_threshold:
            best_iou = iou
            best_match = ann['type_id']
    
    return best_match, best_iou


print("✅ Utility functions loaded!")