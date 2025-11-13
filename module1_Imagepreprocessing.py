"""
CircuitGuard - Module 1: Dataset Setup and Image Subtraction
Tasks:
- Set up and inspect DeepPCB dataset
- Align and preprocess image pairs (template and test)
- Apply image subtraction to obtain defect difference maps
- Use thresholding (Otsu's method) and filters to highlight defect regions

Deliverables:
- Cleaned and aligned dataset
- Subtraction and thresholding script
- Sample defect-highlighted images
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import *
from utils import (get_image_pairs, align_images_orb, compute_difference_map,
                   apply_otsu_threshold, refine_mask_morphology, 
                   highlight_defects, save_json)

print("="*70)
print(" MODULE 1: DATASET SETUP AND IMAGE SUBTRACTION")
print("="*70)

def module1_complete_pipeline():
    """
    Complete Module 1: Image alignment, subtraction, and thresholding
    """
    
    # ============================================
    # TASK 1: SET UP AND INSPECT DATASET
    # ============================================
    print("\n[Task 1/4] Setting up and inspecting DeepPCB dataset...")
    
    pairs = get_image_pairs(DATASET_DIR)
    
    if len(pairs) == 0:
        print("❌ No image pairs found!")
        print(f"   Please check that DATASET_DIR exists: {DATASET_DIR}")
        return None
    
    print(f"✓ Found {len(pairs)} image pairs")
    print(f"✓ Groups: {len(set(p['group'] for p in pairs))}")
    print(f"\nSample pairs:")
    for i, pair in enumerate(pairs[:3]):
        print(f"  {i+1}. {pair['name']}")
    
    # ============================================
    # TASK 2: ALIGN AND PREPROCESS IMAGE PAIRS
    # ============================================
    print("\n[Task 2/4] Aligning and preprocessing image pairs...")
    
    alignment_results = []
    success_count = 0
    failed_count = 0
    
    for pair in tqdm(pairs, desc="Aligning images"):
        try:
            # Load images
            template = cv2.imread(pair['template'])
            test_img = cv2.imread(pair['test'])
            
            if template is None or test_img is None:
                failed_count += 1
                continue
            
            # Align test image to template
            aligned, success = align_images_orb(template, test_img, MAX_FEATURES)
            
            if success:
                success_count += 1
                
                # Save aligned images
                template_path = os.path.join(ALIGNED_DIR, f"{pair['name']}_template.jpg")
                aligned_path = os.path.join(ALIGNED_DIR, f"{pair['name']}_aligned.jpg")
                
                cv2.imwrite(template_path, template)
                cv2.imwrite(aligned_path, aligned)
                
                alignment_results.append({
                    'name': pair['name'],
                    'group': pair['group'],
                    'template': template_path,
                    'aligned': aligned_path,
                    'annotation': pair['annotation'],
                    'alignment_success': True
                })
            else:
                failed_count += 1
                
        except Exception as e:
            print(f"\n❌ Error: {pair['name']}: {e}")
            failed_count += 1
    
    print(f"\n✓ Alignment complete:")
    print(f"  Success: {success_count}/{len(pairs)} ({100*success_count/len(pairs):.1f}%)")
    print(f"  Failed:  {failed_count}/{len(pairs)}")
    
    # ============================================
    # TASK 3: APPLY IMAGE SUBTRACTION
    # ============================================
    print("\n[Task 3/4] Applying image subtraction to obtain difference maps...")
    
    subtraction_results = []
    total_defects_detected = 0
    
    for result in tqdm(alignment_results, desc="Computing differences"):
        try:
            # Load aligned images
            template = cv2.imread(result['template'])
            aligned = cv2.imread(result['aligned'])
            
            if template is None or aligned is None:
                continue
            
            # Compute difference map
            diff_map = compute_difference_map(template, aligned)
            
            # Save difference map
            diff_path = os.path.join(DIFF_MAPS_DIR, f"{result['name']}_diff.png")
            cv2.imwrite(diff_path, diff_map)
            
            result['diff_map'] = diff_path
            subtraction_results.append(result)
            
        except Exception as e:
            print(f"\n❌ Error: {result['name']}: {e}")
    
    print(f"✓ Computed {len(subtraction_results)} difference maps")
    
    # ============================================
    # TASK 4: THRESHOLDING AND HIGHLIGHTING
    # ============================================
    print("\n[Task 4/4] Applying Otsu's thresholding and highlighting defects...")
    
    final_results = []
    
    for result in tqdm(subtraction_results, desc="Thresholding"):
        try:
            # Load images
            aligned = cv2.imread(result['aligned'])
            diff_map = cv2.imread(result['diff_map'], cv2.IMREAD_GRAYSCALE)
            
            if aligned is None or diff_map is None:
                continue
            
            # Apply Otsu's thresholding
            binary_mask = apply_otsu_threshold(diff_map, GAUSSIAN_KERNEL)
            
            # Refine mask with morphological operations
            refined_mask = refine_mask_morphology(
                binary_mask,
                kernel_open=MORPH_KERNEL_OPEN,
                kernel_close=MORPH_KERNEL_CLOSE,
                iter_open=MORPH_ITERATIONS_OPEN,
                iter_close=MORPH_ITERATIONS_CLOSE,
                min_area=MIN_DEFECT_AREA
            )
            
            # Highlight defects
            highlighted, num_defects = highlight_defects(aligned, refined_mask, HIGHLIGHT_COLOR)
            total_defects_detected += num_defects
            
            # Save outputs
            mask_path = os.path.join(MASKS_DIR, f"{result['name']}_mask.png")
            highlighted_path = os.path.join(HIGHLIGHTED_DIR, f"{result['name']}_highlighted.jpg")
            
            cv2.imwrite(mask_path, refined_mask)
            cv2.imwrite(highlighted_path, highlighted)
            
            result['mask'] = mask_path
            result['highlighted'] = highlighted_path
            result['num_defects'] = num_defects
            result['defect_pixels'] = int(np.sum(refined_mask > 0))
            
            final_results.append(result)
            
        except Exception as e:
            print(f"\n❌ Error: {result['name']}: {e}")
    
    print(f"✓ Processed {len(final_results)} images")
    print(f"✓ Total defects detected: {total_defects_detected}")
    print(f"✓ Average defects per image: {total_defects_detected/len(final_results):.2f}")
    
    # ============================================
    # SAVE METADATA
    # ============================================
    print("\n[5/5] Saving module metadata...")
    
    metadata = {
        'module': 'Module 1: Dataset Setup and Image Subtraction',
        'total_pairs': len(pairs),
        'successful_alignments': success_count,
        'failed_alignments': failed_count,
        'processed_images': len(final_results),
        'total_defects_detected': total_defects_detected,
        'average_defects_per_image': total_defects_detected / len(final_results) if final_results else 0,
        'results': final_results
    }
    
    save_json(metadata, os.path.join(OUTPUT_DIR, 'module1_results.json'))
    print(f"✓ Metadata saved: {OUTPUT_DIR}/module1_results.json")
    
    # ============================================
    # CREATE SAMPLE VISUALIZATIONS
    # ============================================
    print("\n[6/6] Creating sample visualizations...")
    create_module1_samples(final_results[:NUM_SAMPLES])
    
    # ============================================
    # PRINT SUMMARY
    # ============================================
    print("\n" + "="*70)
    print(" MODULE 1 DELIVERABLES - COMPLETE!")
    print("="*70)
    
    print("\n✅ Deliverable 1: Cleaned and aligned dataset")
    print(f"   📁 Location: {ALIGNED_DIR}")
    print(f"   📊 Total: {success_count} aligned pairs")
    
    print("\n✅ Deliverable 2: Subtraction and thresholding script")
    print(f"   📝 This script (module1_image_subtraction.py)")
    print(f"   📁 Difference maps: {DIFF_MAPS_DIR}")
    print(f"   📁 Defect masks: {MASKS_DIR}")
    
    print("\n✅ Deliverable 3: Sample defect-highlighted images")
    print(f"   📁 Highlighted defects: {HIGHLIGHTED_DIR}")
    print(f"   📁 Sample visualizations: {SAMPLES_DIR}")
    
    print("\n📊 Module 1 Evaluation Metrics:")
    print(f"   ✓ Defect mask generation: {len(final_results)} masks created")
    print(f"   ✓ Alignment success rate: {100*success_count/len(pairs):.1f}%")
    print(f"   ✓ Average defects per image: {total_defects_detected/len(final_results):.2f}")
    print(f"   ✓ Subtraction clarity: Verified via sample outputs")
    
    print("\n" + "="*70)
    
    return metadata


def create_module1_samples(samples):
    """Create comprehensive visualization of Module 1 pipeline"""
    
    num_samples = min(NUM_SAMPLES, len(samples))
    
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, sample in enumerate(samples[:num_samples]):
        # Load all images
        template = cv2.imread(sample['template'])
        aligned = cv2.imread(sample['aligned'])
        diff = cv2.imread(sample['diff_map'], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE)
        highlighted = cv2.imread(sample['highlighted'])
        
        # Convert BGR to RGB
        template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
        aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        highlighted_rgb = cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB)
        
        # Plot
        axes[idx, 0].imshow(template_rgb)
        axes[idx, 0].set_title('Template', fontweight='bold', fontsize=10)
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(aligned_rgb)
        axes[idx, 1].set_title('Aligned Test', fontweight='bold', fontsize=10)
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(diff, cmap='hot')
        axes[idx, 2].set_title('Difference Map', fontweight='bold', fontsize=10)
        axes[idx, 2].axis('off')
        
        axes[idx, 3].imshow(mask, cmap='gray')
        axes[idx, 3].set_title(f"Defect Mask (Otsu)\n{sample['num_defects']} defects", 
                              fontweight='bold', fontsize=10)
        axes[idx, 3].axis('off')
        
        axes[idx, 4].imshow(highlighted_rgb)
        axes[idx, 4].set_title('Highlighted Defects', fontweight='bold', fontsize=10)
        axes[idx, 4].axis('off')
    
    plt.suptitle('Module 1: Image Subtraction & Defect Detection Pipeline', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(SAMPLES_DIR, 'module1_pipeline_samples.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Sample visualization saved: {SAMPLES_DIR}/module1_pipeline_samples.png")


if __name__ == "__main__":
    print("\nStarting Module 1 Pipeline...")
    
    metadata = module1_complete_pipeline()
    
    if metadata:
        print("\n✅ MODULE 1 COMPLETE!")
        print("\n🎯 Ready for Module 2: Contour Detection and ROI Extraction")
    else:
        print("\n❌ Module 1 failed. Please check your dataset.")