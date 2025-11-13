"""
CircuitGuard - Module 5: Enhanced Web UI for Image Upload and Visualization
Professional Streamlit-based frontend for PCB defect detection
"""

import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import time

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="CircuitGuard - AI PCB Inspector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# ENHANCED CUSTOM CSS
# ============================================
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: #f0f0f0;
        font-size: 1.3rem;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Metric boxes */
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .metric-box h2 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: bold;
    }
    
    .metric-box p {
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Defect type badge */
    .defect-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1rem;
        margin: 0.3rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.3rem;
        font-weight: 600;
        padding: 0.8rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Upload area styling */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Success/info boxes */
    .element-container div.stAlert {
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    
    /* Image containers */
    .image-container {
        border: 3px solid #667eea;
        border-radius: 12px;
        padding: 0.5rem;
        background: white;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# CONFIGURATION
# ============================================
MODEL_PATH = r"C:\CircuitGuardd_Infosyss\milestone2_output\models\mobilenetv2_best.pth"
IMAGE_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFECT_TYPES = {
    0: 'mousebite',
    1: 'open',
    2: 'pin_hole',
    3: 'short',
    4: 'spur',
    5: 'spurious_copper'
}

DEFECT_COLORS = {
    'mousebite': '#FF6B6B',
    'open': '#4ECDC4',
    'pin_hole': '#45B7D1',
    'short': '#FFA07A',
    'spur': '#98D8C8',
    'spurious_copper': '#F7DC6F'
}

DEFECT_COLORS_BGR = {
    'mousebite': (107, 107, 255),
    'open': (196, 205, 78),
    'pin_hole': (209, 183, 69),
    'short': (122, 160, 255),
    'spur': (200, 216, 152),
    'spurious_copper': (111, 220, 247)
}

# ============================================
# LOAD MODEL (CACHED)
# ============================================
@st.cache_resource
def load_model():
    """Load trained model"""
    try:
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.last_channel, len(DEFECT_TYPES))
        
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(DEVICE)
        model.eval()
        
        return model, checkpoint
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

# ============================================
# IMAGE PROCESSING FUNCTIONS
# ============================================
def align_images(template, test_img):
    """Align test image to template"""
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    template_gray = clahe.apply(template_gray)
    test_gray = clahe.apply(test_gray)
    
    orb = cv2.ORB_create(1000)
    kp1, desc1 = orb.detectAndCompute(template_gray, None)
    kp2, desc2 = orb.detectAndCompute(test_gray, None)
    
    if desc1 is None or desc2 is None:
        return test_img, False
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)
    
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < 10:
        return test_img, False
    
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        return test_img, False
    
    h, w = template.shape[:2]
    aligned = cv2.warpPerspective(test_img, H, (w, h))
    
    return aligned, True

def detect_defects(template, test_img):
    """Detect defects using subtraction"""
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(template_gray, test_gray)
    blurred = cv2.GaussianBlur(diff, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return closed

def extract_rois(image, mask, min_area=30, padding=15):
    """Extract ROIs from mask"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rois = []
    h, w = image.shape[:2]
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        x, y, box_w, box_h = cv2.boundingRect(contour)
        
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + box_w + padding)
        y2 = min(h, y + box_h + padding)
        
        roi_img = image[y1:y2, x1:x2]
        
        if roi_img.size > 0:
            rois.append({
                'image': roi_img,
                'bbox': (x1, y1, x2, y2)
            })
    
    return rois

def classify_roi(model, roi_image):
    """Classify ROI"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    roi_rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
    roi_tensor = transform(roi_rgb).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(roi_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item()

def draw_predictions(image, predictions):
    """Draw predictions on image with better label positioning"""
    result = image.copy()
    h, w = result.shape[:2]
    
    # Sort predictions by y-coordinate to avoid overlap
    sorted_preds = sorted(predictions, key=lambda p: p['bbox'][1])
    
    used_positions = []
    
    for pred in sorted_preds:
        bbox = pred['bbox']
        defect_type = pred['defect_type']
        confidence = pred['confidence']
        
        color = DEFECT_COLORS_BGR.get(defect_type, (255, 255, 255))
        
        # Draw bounding box with thicker line
        cv2.rectangle(result, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Draw filled circle at top-left corner
        cv2.circle(result, (bbox[0], bbox[1]), 8, color, -1)
        
        # Create label
        label = f"{defect_type}: {confidence:.0%}"
        font_scale = 0.5
        thickness = 1
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                                        font_scale, thickness)
        
        # Find non-overlapping position for label
        label_x = bbox[0]
        label_y = bbox[1] - 10
        
        # Adjust if label goes out of bounds
        if label_y - label_h < 0:
            label_y = bbox[3] + label_h + 10
        if label_x + label_w > w:
            label_x = w - label_w - 10
        
        # Draw label background
        cv2.rectangle(result, (label_x - 2, label_y - label_h - 4), 
                     (label_x + label_w + 2, label_y + 4), color, -1)
        
        # Draw label text
        cv2.putText(result, label, (label_x, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    return result

# ============================================
# MAIN APPLICATION
# ============================================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔬 CircuitGuard AI</h1>
        <p>Advanced PCB Defect Detection System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, checkpoint = load_model()
    
    if model is None:
        st.error("⚠️ Failed to load model. Please check MODEL_PATH.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 System Status")
        st.success("🟢 Model Loaded")
        
        st.markdown("---")
        st.markdown("### 🎯 Model Information")
        st.markdown(f"""
        <div class="info-card">
            <strong>Architecture:</strong> MobileNetV2<br>
            <strong>Accuracy:</strong> {checkpoint['val_acc']:.2f}%<br>
            <strong>Epoch:</strong> {checkpoint['epoch']}<br>
            <strong>Device:</strong> {str(DEVICE).upper()}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🏷️ Detectable Defects")
        
        for defect_type, color in DEFECT_COLORS.items():
            st.markdown(f"""
            <div class="defect-badge" style="background-color:{color}; color:white;">
                {defect_type}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. 📤 Upload template image (defect-free PCB)
        2. 📤 Upload test image (PCB to inspect)
        3. 🔍 Click "Start Detection"
        4. 📊 View detailed results
        """)
    
    # Main content
    st.markdown("### 📤 Upload Images")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        template_file = st.file_uploader("🖼️ Template Image (Defect-Free)", 
                                         type=['jpg', 'jpeg', 'png'],
                                         key="template")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        test_file = st.file_uploader("🖼️ Test Image (To Inspect)", 
                                     type=['jpg', 'jpeg', 'png'],
                                     key="test")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display uploaded images
    if template_file and test_file:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(template_file, caption="✅ Template Image", use_container_width=True)
        
        with col2:
            st.image(test_file, caption="🔍 Test Image", use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Centered detect button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            detect_button = st.button("🚀 Start Defect Detection", type="primary")
        
        if detect_button:
            with st.spinner("🔄 Processing..."):
                # Read images
                template_bytes = np.asarray(bytearray(template_file.read()), dtype=np.uint8)
                test_bytes = np.asarray(bytearray(test_file.read()), dtype=np.uint8)
                
                template_img = cv2.imdecode(template_bytes, cv2.IMREAD_COLOR)
                test_img = cv2.imdecode(test_bytes, cv2.IMREAD_COLOR)
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Align
                status_text.info("⏳ Step 1/4: Aligning images...")
                aligned_img, success = align_images(template_img, test_img)
                progress_bar.progress(25)
                time.sleep(0.3)
                
                if not success:
                    st.warning("⚠️ Alignment confidence low. Using original image.")
                    aligned_img = test_img
                
                # Step 2: Detect
                status_text.info("⏳ Step 2/4: Detecting defects...")
                mask = detect_defects(template_img, aligned_img)
                progress_bar.progress(50)
                time.sleep(0.3)
                
                # Step 3: Extract ROIs
                status_text.info("⏳ Step 3/4: Extracting defect regions...")
                rois = extract_rois(aligned_img, mask)
                progress_bar.progress(75)
                time.sleep(0.3)
                
                if len(rois) == 0:
                    progress_bar.progress(100)
                    status_text.empty()
                    st.success("✨ Perfect! No defects detected. PCB is clean!")
                    st.balloons()
                else:
                    # Step 4: Classify
                    status_text.info(f"⏳ Step 4/4: Classifying {len(rois)} defect(s)...")
                    
                    predictions = []
                    for roi in rois:
                        pred_class, confidence = classify_roi(model, roi['image'])
                        defect_type = DEFECT_TYPES[pred_class]
                        
                        predictions.append({
                            'bbox': roi['bbox'],
                            'defect_type': defect_type,
                            'confidence': confidence
                        })
                    
                    progress_bar.progress(100)
                    status_text.empty()
                    
                    # Draw results
                    result_img = draw_predictions(aligned_img, predictions)
                    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    
                    # Success message
                    st.success(f"✅ Analysis Complete! Detected {len(predictions)} defect(s)")
                    
                    # Summary metrics
                    st.markdown("### 📊 Detection Summary")
                    
                    defect_summary = {}
                    for pred in predictions:
                        dt = pred['defect_type']
                        defect_summary[dt] = defect_summary.get(dt, 0) + 1
                    
                    cols = st.columns(len(defect_summary))
                    for idx, (defect_type, count) in enumerate(defect_summary.items()):
                        with cols[idx]:
                            color = DEFECT_COLORS[defect_type]
                            st.markdown(f"""
                            <div class="metric-box" style="background: {color};">
                                <h2>{count}</h2>
                                <p>{defect_type.upper()}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Annotated result
                    st.markdown("### 🎯 Annotated Results")
                    st.image(result_rgb, caption="Defects Highlighted", use_container_width=True)
                    
                    # Detailed predictions
                    st.markdown("### 📋 Detailed Analysis")
                    
                    for idx, pred in enumerate(predictions, 1):
                        color = DEFECT_COLORS[pred['defect_type']]
                        bbox = pred['bbox']
                        
                        with st.expander(f"🔍 Defect #{idx}: {pred['defect_type'].upper()} - {pred['confidence']:.1%} confidence"):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                roi_crop = aligned_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                                roi_rgb = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB)
                                st.image(roi_rgb, caption=f"Defect Region", width=300)
                            
                            with col2:
                                st.markdown(f"""
                                <div style="padding:1rem; background:{color}; color:white; border-radius:10px;">
                                    <h3 style="margin:0;">{pred['defect_type'].upper()}</h3>
                                    <p style="font-size:1.5rem; margin:0.5rem 0;"><strong>{pred['confidence']:.1%}</strong></p>
                                    <p style="margin:0; font-size:0.9rem;">
                                        Location: ({bbox[0]}, {bbox[1]})<br>
                                        Size: {bbox[2]-bbox[0]} x {bbox[3]-bbox[1]}px
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Download button
                    st.markdown("### 💾 Export Results")
                    _, buffer = cv2.imencode('.jpg', result_img)
                    st.download_button(
                        label="⬇️ Download Annotated Image",
                        data=buffer.tobytes(),
                        file_name=f"circuitguard_result_{int(time.time())}.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )
    
    else:
        st.info("👆 Please upload both template and test images to begin defect detection.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 10px; color: white;'>
        <h3 style='margin:0;'>CircuitGuard AI - PCB Defect Detection System</h3>
        <p style='margin:0.5rem 0 0 0; opacity:0.9;'>Powered by MobileNetV2 Deep Learning | Milestone 3 Complete</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()