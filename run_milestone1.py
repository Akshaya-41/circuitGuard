"""
CircuitGuard - Milestone 1 Complete Runner
Runs both Module 1 and Module 2 in sequence
"""

import sys
import time
from datetime import datetime

print("="*70)
print(" CIRCUITGUARD - MILESTONE 1")
print(" Dataset Preparation and Image Processing")
print("="*70)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

total_start = time.time()

# ============================================
# MODULE 1: DATASET SETUP & IMAGE SUBTRACTION
# ============================================
print("\n\n" + "█"*70)
print("█  MODULE 1: DATASET SETUP AND IMAGE SUBTRACTION")
print("█"*70 + "\n")

module1_start = time.time()

try:
    # Import and run Module 1
    import module1_Imagepreprocessing
    result1 = module1_Imagepreprocessing.module1_complete_pipeline()
    
    if not result1:
        print("\n❌ Module 1 failed! Stopping pipeline.")
        sys.exit(1)
    
    module1_time = time.time() - module1_start
    print(f"\n✅ Module 1 completed in {module1_time/60:.2f} minutes")
    
except Exception as e:
    print(f"\n❌ Module 1 Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================
# MODULE 2: CONTOUR DETECTION & ROI EXTRACTION
# ============================================
print("\n\n" + "█"*70)
print("█  MODULE 2: CONTOUR DETECTION AND ROI EXTRACTION")
print("█"*70 + "\n")

module2_start = time.time()

try:
    # Import and run Module 2
    import module2_contour_roiExtraction
    result2 = module2_contour_roiExtraction.module2_complete_pipeline()
    
    if not result2:
        print("\n❌ Module 2 failed! Check module2 output.")
        sys.exit(1)
    
    module2_time = time.time() - module2_start
    print(f"\n✅ Module 2 completed in {module2_time/60:.2f} minutes")
    
except Exception as e:
    print(f"\n❌ Module 2 Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================
# FINAL SUMMARY
# ============================================
total_time = time.time() - total_start

print("\n\n" + "="*70)
print(" MILESTONE 1 COMPLETE - FINAL SUMMARY")
print("="*70)

print(f"\n⏱ Total Time: {total_time/60:.2f} minutes")
print(f"   Module 1: {module1_time/60:.2f} minutes")
print(f"   Module 2: {module2_time/60:.2f} minutes")

print("\n📊 Results Summary:")
print(f"   ✓ Images aligned: {result1['successful_alignments']}")
print(f"   ✓ Defects detected: {result1['total_defects_detected']}")
print(f"   ✓ Contours found: {result2['total_contours_detected']}")
print(f"   ✓ ROIs extracted: {result2['total_rois_extracted']}")
print(f"   ✓ Labeled ROIs: {result2['matched_rois']}")

print("\n📁 Output Location:")
from config import OUTPUT_DIR
print(f"   {OUTPUT_DIR}")

print("\n✅ MILESTONE 1 DELIVERABLES:")
print("\n   Module 1:")
print("   ✓ Cleaned and aligned dataset")
print("   ✓ Subtraction and thresholding script")
print("   ✓ Sample defect-highlighted images")

print("\n   Module 2:")
print("   ✓ ROI extraction pipeline")
print("   ✓ Cropped and labeled defect samples")
print("   ✓ Visualization of defect contours")

print("\n📈 Evaluation Metrics:")
print(f"   ✓ Alignment success rate: {100*result1['successful_alignments']/result1['total_pairs']:.1f}%")
print(f"   ✓ ROI match rate: {100*result2['match_rate']:.1f}%")
print(f"   ✓ Average defects per image: {result1['average_defects_per_image']:.2f}")

print("\n🎯 Next Steps:")
print("   → Review sample outputs in milestone1_output/sample_outputs/")
print("   → Verify ROI quality in milestone1_output/extracted_rois/")
print("   → Ready to proceed to Milestone 2: Model Training")

print("\n" + "="*70)
print(f" End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)