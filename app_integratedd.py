"""
CircuitGuard - Integrated Frontend with Backend Pipeline
Complete Module 5 & 6 Integration with PDF Report Generation
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import io
from datetime import datetime

# Import backend pipeline
from backend_pipeline import InferencePipeline, BackendConfig

# For PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.warning("⚠️ Install reportlab and matplotlib for PDF report: pip install reportlab matplotlib")

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
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
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
    
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
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
    
    .defect-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1rem;
        margin: 0.3rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
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
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .log-box {
        background: #2d3748;
        color: #48bb78;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #4a5568;
    }
    
    .time-badge {
        display: inline-block;
        background: #48bb78;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# INITIALIZE BACKEND
# ============================================
@st.cache_resource
def load_backend():
    """Load backend pipeline (cached)"""
    return InferencePipeline(BackendConfig.MODEL_PATH)

# ============================================
# PDF REPORT GENERATOR
# ============================================
def generate_pdf_report(results, template_img, test_img):
    """Generate comprehensive PDF report with charts"""
    
    if not PDF_AVAILABLE:
        return None
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#764ba2'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    story.append(Paragraph("CircuitGuard AI - Defect Detection Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Timestamp
    story.append(Paragraph(f"<b>Generated:</b> {results['timestamp']}", styles['Normal']))
    story.append(Paragraph(f"<b>Processing Time:</b> {results['processing_time']['total']:.2f} seconds", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Summary Section
    story.append(Paragraph("Executive Summary", heading_style))
    summary_data = [
        ['Metric', 'Value'],
        ['Total Defects Detected', str(results['num_defects'])],
        ['Alignment Status', '✓ Successful' if results['alignment_success'] else '✗ Failed'],
        ['Total Processing Time', f"{results['processing_time']['total']:.2f}s"],
        ['Classification Time', f"{results['processing_time']['classification']:.2f}s"],
    ]
    
    summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Defect Distribution
    if results['num_defects'] > 0:
        story.append(Paragraph("Defect Analysis", heading_style))
        
        # Count defects by type
        defect_counts = {}
        for pred in results['predictions']:
            dt = pred['defect_type']
            defect_counts[dt] = defect_counts.get(dt, 0) + 1
        
        # Create bar chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Bar chart
        defect_types = list(defect_counts.keys())
        counts = list(defect_counts.values())
        colors_chart = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
        
        ax1.bar(defect_types, counts, color=colors_chart[:len(defect_types)])
        ax1.set_xlabel('Defect Type', fontweight='bold')
        ax1.set_ylabel('Count', fontweight='bold')
        ax1.set_title('Defect Distribution (Bar Chart)', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # Pie chart
        ax2.pie(counts, labels=defect_types, autopct='%1.1f%%', startangle=90,
               colors=colors_chart[:len(defect_types)])
        ax2.set_title('Defect Distribution (Pie Chart)', fontweight='bold')
        
        plt.tight_layout()
        
        # Save chart to buffer
        chart_buffer = io.BytesIO()
        plt.savefig(chart_buffer, format='png', dpi=150, bbox_inches='tight')
        chart_buffer.seek(0)
        plt.close()
        
        # Add chart to PDF
        chart_img = RLImage(chart_buffer, width=6*inch, height=2.4*inch)
        story.append(chart_img)
        story.append(Spacer(1, 0.3*inch))
        
        # Defect details table
        story.append(Paragraph("Detailed Defect Information", heading_style))
        defect_data = [['#', 'Type', 'Confidence', 'Location (x, y)', 'Size (px²)']]
        
        for idx, pred in enumerate(results['predictions'], 1):
            bbox = pred['bbox']
            defect_data.append([
                str(idx),
                pred['defect_type'],
                f"{pred['confidence']:.1%}",
                f"({bbox[0]}, {bbox[1]})",
                str(pred['area'])
            ])
        
        defect_table = Table(defect_data, colWidths=[0.5*inch, 1.5*inch, 1*inch, 1.5*inch, 1*inch])
        defect_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        story.append(defect_table)
    
    else:
        story.append(Paragraph("✓ No defects detected! PCB appears to be defect-free.", styles['Normal']))
    
    story.append(PageBreak())
    
    # Annotated Image
    story.append(Paragraph("Annotated Image", heading_style))
    
    # Convert annotated image to RGB and save
    annotated_rgb = cv2.cvtColor(results['annotated_image'], cv2.COLOR_BGR2RGB)
    img_buffer = io.BytesIO()
    Image.fromarray(annotated_rgb).save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    # Add to PDF (scaled to fit)
    annotated_img = RLImage(img_buffer, width=6*inch, height=4*inch)
    story.append(annotated_img)
    story.append(Spacer(1, 0.3*inch))
    
    # Processing Logs
    story.append(PageBreak())
    story.append(Paragraph("Processing Logs", heading_style))
    
    for log in results['logs']:
        story.append(Paragraph(log, styles['Code']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    return buffer

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
    
    # Load backend
    try:
        pipeline = load_backend()
        backend_loaded = True
    except Exception as e:
        st.error(f"⚠️ Failed to load backend: {e}")
        backend_loaded = False
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 System Status")
        st.success("🟢 Backend Loaded")
        st.success("🟢 Model Ready")
        
        st.markdown("---")
        st.markdown("### 🎯 Model Information")
        st.markdown(f"""
        <div class="info-card">
            <strong>Architecture:</strong> MobileNetV2<br>
            <strong>Accuracy:</strong> {pipeline.model_manager.checkpoint['val_acc']:.2f}%<br>
            <strong>Device:</strong> {str(BackendConfig.DEVICE).upper()}<br>
            <strong>Image Size:</strong> {BackendConfig.IMAGE_SIZE}x{BackendConfig.IMAGE_SIZE}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🏷️ Detectable Defects")
        
        defect_colors = {
            'mousebite': '#FF6B6B',
            'open': '#4ECDC4',
            'pin_hole': '#45B7D1',
            'short': '#FFA07A',
            'spur': '#98D8C8',
            'spurious_copper': '#F7DC6F'
        }
        
        for defect_type, color in defect_colors.items():
            st.markdown(f"""
            <div class="defect-badge" style="background-color:{color}; color:white;">
                {defect_type}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. 📤 Upload template image
        2. 📤 Upload test image
        3. 🔍 Click "Start Detection"
        4. 📊 View results and logs
        """)
    
    # Main content
    st.markdown("### 📤 Upload Images")
    
    col1, col2 = st.columns(2)
    
    with col1:
        template_file = st.file_uploader("🖼️ Template Image (Defect-Free)", 
                                         type=['jpg', 'jpeg', 'png'],
                                         key="template")
    
    with col2:
        test_file = st.file_uploader("🖼️ Test Image (To Inspect)", 
                                     type=['jpg', 'jpeg', 'png'],
                                     key="test")
    
    # Display uploaded images
    if template_file and test_file:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(template_file, caption="✅ Template Image", use_container_width=True)
        
        with col2:
            st.image(test_file, caption="🔍 Test Image", use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Detect button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            detect_button = st.button("🚀 Start Defect Detection", type="primary")
        
        if detect_button:
            # Read images
            template_bytes = np.asarray(bytearray(template_file.read()), dtype=np.uint8)
            test_bytes = np.asarray(bytearray(test_file.read()), dtype=np.uint8)
            
            template_img = cv2.imdecode(template_bytes, cv2.IMREAD_COLOR)
            test_img = cv2.imdecode(test_bytes, cv2.IMREAD_COLOR)
            
            # Process with backend
            with st.spinner("🔄 Processing with backend pipeline..."):
                results = pipeline.process_images(template_img, test_img)
            
            if results['success']:
                # Display processing logs
                st.markdown("### 📋 Processing Logs")
                log_text = "\n".join(results['logs'])
                st.markdown(f'<div class="log-box">{log_text}</div>', unsafe_allow_html=True)
                
                # Display processing time
                st.markdown("### ⏱️ Performance Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <h2>{results['processing_time']['alignment']:.2f}s</h2>
                        <p>Alignment</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-box">
                        <h2>{results['processing_time']['detection']:.2f}s</h2>
                        <p>Detection</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-box">
                        <h2>{results['processing_time']['classification']:.2f}s</h2>
                        <p>Classification</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-box">
                        <h2>{results['processing_time']['total']:.2f}s</h2>
                        <p>Total Time</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Check if defects found
                if results['num_defects'] == 0:
                    st.success("✨ Perfect! No defects detected. PCB is clean!")
                    st.balloons()
                else:
                    st.success(f"✅ Detection complete! Found {results['num_defects']} defect(s)")
                    
                    # Summary metrics
                    st.markdown("### 📊 Detection Summary")
                    
                    defect_summary = {}
                    for pred in results['predictions']:
                        dt = pred['defect_type']
                        defect_summary[dt] = defect_summary.get(dt, 0) + 1
                    
                    cols = st.columns(len(defect_summary))
                    for idx, (defect_type, count) in enumerate(defect_summary.items()):
                        with cols[idx]:
                            color = defect_colors.get(defect_type, '#667eea')
                            st.markdown(f"""
                            <div class="metric-box" style="background: {color};">
                                <h2>{count}</h2>
                                <p>{defect_type.upper()}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Annotated result (REDUCED SIZE)
                    st.markdown("### 🎯 Annotated Results")
                    result_rgb = cv2.cvtColor(results['annotated_image'], cv2.COLOR_BGR2RGB)
                    
                    # Display at 60% width
                    col1, col2, col3 = st.columns([1, 3, 1])
                    with col2:
                        st.image(result_rgb, caption="Defects Highlighted", use_container_width=True)
                    
                    # Detailed predictions
                    st.markdown("### 📋 Detailed Analysis")
                    
                    for idx, pred in enumerate(results['predictions'], 1):
                        color = defect_colors.get(pred['defect_type'], '#667eea')
                        bbox = pred['bbox']
                        
                        with st.expander(f"🔍 Defect #{idx}: {pred['defect_type'].upper()} - {pred['confidence']:.1%} confidence"):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                roi_crop = results['aligned_image'][bbox[1]:bbox[3], bbox[0]:bbox[2]]
                                roi_rgb = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB)
                                st.image(roi_rgb, caption=f"Defect Region", width=300)
                            
                            with col2:
                                st.markdown(f"""
                                <div style="padding:1rem; background:{color}; color:white; border-radius:10px;">
                                    <h3 style="margin:0;">{pred['defect_type'].upper()}</h3>
                                    <p style="font-size:1.5rem; margin:0.5rem 0;"><strong>{pred['confidence']:.1%}</strong></p>
                                    <p style="margin:0; font-size:0.9rem;">
                                        <strong>Location:</strong> ({bbox[0]}, {bbox[1]})<br>
                                        <strong>Size:</strong> {bbox[2]-bbox[0]} x {bbox[3]-bbox[1]}px<br>
                                        <strong>Area:</strong> {pred['area']} px²
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Download options
                    st.markdown("### 💾 Download Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Download PDF Report
                        if PDF_AVAILABLE:
                            pdf_buffer = generate_pdf_report(results, template_img, test_img)
                            if pdf_buffer:
                                st.download_button(
                                    label="📄 Download Complete Report (PDF)",
                                    data=pdf_buffer,
                                    file_name=f"CircuitGuard_Report_{results['timestamp'].replace(' ', '_').replace(':', '-')}.pdf",
                                    mime="application/pdf",
                                    use_container_width=True
                                )
                            else:
                                st.error("Failed to generate PDF report")
                        else:
                            st.warning("Install reportlab for PDF reports: pip install reportlab matplotlib")
                    
                    with col2:
                        # Download logs
                        log_content = "\n".join(results['logs'])
                        st.download_button(
                            label="📋 Download Processing Logs",
                            data=log_content,
                            file_name=f"CircuitGuard_Logs_{results['timestamp'].replace(' ', '_').replace(':', '-')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
            
            else:
                st.error(f"❌ Processing failed: {results.get('error', 'Unknown error')}")
                if 'logs' in results:
                    st.markdown("### 📋 Error Logs")
                    log_text = "\n".join(results['logs'])
                    st.markdown(f'<div class="log-box">{log_text}</div>', unsafe_allow_html=True)
    
    else:
        st.info("👆 Please upload both template and test images to begin defect detection.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 10px; color: white;'>
        <h3 style='margin:0;'>CircuitGuard AI - PCB Defect Detection System</h3>
        <p style='margin:0.5rem 0 0 0; opacity:0.9;'>Powered by MobileNetV2 | Backend Pipeline Integration Complete</p>
        <p style='margin:0.3rem 0 0 0; font-size:0.9rem;'>Module 5 & 6 Complete ✅</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()