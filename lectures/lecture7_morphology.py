import streamlit as st
import cv2
import numpy as np
from utils.image_utils import load_image, display_images
from utils.processing import apply_threshold, morphological_operation

def show():
    st.header("المحاضرة السابعة: العمليات المورفولوجية")
    
    # Theory section
    st.subheader("النظرية")
    st.write("""
    العمليات المورفولوجية تعالج الصور بناءً على الشكل باستخدام العناصر البنيوية (النوى).
    التآكل يقلص المناطق البيضاء، مفيد لإزالة الضوضاء وفصل الأشياء المتصلة.
    التمدد يوسع المناطق البيضاء، مفيد لملء الثقوب وربط الخطوط المكسورة.
    الفتح (تآكل ثم تمدد) يزيل الضوضاء الصغيرة مع الحفاظ على البنى الأكبر.
    الإغلاق (تمدد ثم تآكل) يملأ الفجوات والثقوب مع الحفاظ على حجم الكائن.
    """)
    
    st.subheader("العرض التفاعلي")
    
    # Load image
    image = load_image()
    
    if image is not None:
        # Convert to grayscale first
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Create binary image for morphology
        st.subheader("الخطوة 1: إنشاء صورة ثنائية")
        
        threshold_value = st.slider("عتبة التحويل الثنائي", 0, 255, 127)
        
        # Apply thresholding
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("الرمادية الأصلية")
            st.image(gray, use_container_width=True)
        
        with col2:
            st.subheader("الصورة الثنائية")
            st.image(binary, use_container_width=True)
        
        st.write(f"العتبة: {threshold_value} | البكسلات البيضاء: {np.sum(binary == 255):,}")
        
        # Morphological operations
        st.subheader("الخطوة 2: تطبيق العمليات المورفولوجية")
        
        operation = st.selectbox(
            "اختر العملية المورفولوجية:",
            ["التآكل", "التمدد", "الفتح", "الإغلاق", "مقارنة الكل"]
        )
        
        if operation != "Compare All":
            col1, col2 = st.columns(2)
            
            with col1:
                kernel_size = st.slider("Kernel Size", 3, 15, 5, 2)
            
            with col2:
                if operation in ["Erosion", "Dilation"]:
                    iterations = st.slider("Iterations", 1, 10, 1)
                else:
                    iterations = 1
                    st.write("Single iteration for opening/closing")
            
            # Show kernel visualization
            if st.checkbox("Show Structuring Element"):
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                
                # Create a larger visualization of the kernel
                kernel_vis = np.repeat(np.repeat(kernel * 255, 20, axis=0), 20, axis=1)
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.write(f"**{kernel_size}×{kernel_size} Kernel:**")
                    st.image(kernel_vis, width=150)
                
                with col2:
                    st.write("**Properties:**")
                    st.write(f"Shape: Rectangle")
                    st.write(f"Size: {kernel_size} × {kernel_size}")
                    st.write(f"Total elements: {kernel_size * kernel_size}")
                    st.write("All elements are 1 (structuring)")
            
            # Apply selected operation
            if operation == "Erosion":
                result = cv2.erode(binary, np.ones((kernel_size, kernel_size), np.uint8), iterations=iterations)
                
                display_images(binary, result, ["Binary Input", f"Eroded (k={kernel_size}, i={iterations})"])
                
                st.write("**Erosion Effect:**")
                st.write("- Shrinks white (foreground) regions")
                st.write("- Removes small noise and thin connections")
                st.write("- Can separate touching objects")
                
                # Quantitative analysis
                original_white = np.sum(binary == 255)
                result_white = np.sum(result == 255)
                reduction = ((original_white - result_white) / original_white) * 100
                
                st.write(f"**Pixel Reduction:** {reduction:.1f}% ({original_white:,} → {result_white:,})")
            
            elif operation == "Dilation":
                result = cv2.dilate(binary, np.ones((kernel_size, kernel_size), np.uint8), iterations=iterations)
                
                display_images(binary, result, ["Binary Input", f"Dilated (k={kernel_size}, i={iterations})"])
                
                st.write("**Dilation Effect:**")
                st.write("- Expands white (foreground) regions")
                st.write("- Fills small holes and gaps")
                st.write("- Connects nearby objects")
                
                # Quantitative analysis
                original_white = np.sum(binary == 255)
                result_white = np.sum(result == 255)
                increase = ((result_white - original_white) / original_white) * 100
                
                st.write(f"**Pixel Increase:** {increase:.1f}% ({original_white:,} → {result_white:,})")
            
            elif operation == "Opening":
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                
                display_images(binary, result, ["Binary Input", f"Opened (k={kernel_size})"])
                
                st.write("**Opening Effect (Erosion → Dilation):**")
                st.write("- Removes small noise while preserving large objects")
                st.write("- Smooths object contours")
                st.write("- Separates touching objects")
                
                # Show intermediate steps
                if st.checkbox("Show Intermediate Steps"):
                    eroded = cv2.erode(binary, kernel, iterations=1)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**1. Original**")
                        st.image(binary, use_column_width=True)
                    
                    with col2:
                        st.write("**2. After Erosion**")
                        st.image(eroded, use_column_width=True)
                    
                    with col3:
                        st.write("**3. After Dilation (Final)**")
                        st.image(result, use_column_width=True)
            
            elif operation == "Closing":
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                
                display_images(binary, result, ["Binary Input", f"Closed (k={kernel_size})"])
                
                st.write("**Closing Effect (Dilation → Erosion):**")
                st.write("- Fills holes and gaps in objects")
                st.write("- Connects broken lines")
                st.write("- Smooths object boundaries")
                
                # Show intermediate steps
                if st.checkbox("Show Intermediate Steps"):
                    dilated = cv2.dilate(binary, kernel, iterations=1)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**1. Original**")
                        st.image(binary, use_column_width=True)
                    
                    with col2:
                        st.write("**2. After Dilation**")
                        st.image(dilated, use_column_width=True)
                    
                    with col3:
                        st.write("**3. After Erosion (Final)**")
                        st.image(result, use_column_width=True)
        
        else:  # Compare All
            st.subheader("🔍 Morphological Operations Comparison")
            
            kernel_size_comp = st.slider("Kernel Size for Comparison", 3, 11, 5, 2)
            kernel = np.ones((kernel_size_comp, kernel_size_comp), np.uint8)
            
            # Apply all operations
            eroded = cv2.erode(binary, kernel, iterations=1)
            dilated = cv2.dilate(binary, kernel, iterations=1)
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Display all results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Original Binary")
                st.image(binary, use_column_width=True)
                white_orig = np.sum(binary == 255)
                st.write(f"White pixels: {white_orig:,}")
            
            with col2:
                st.subheader("Erosion")
                st.image(eroded, use_column_width=True)
                white_eroded = np.sum(eroded == 255)
                change_eroded = ((white_eroded - white_orig) / white_orig) * 100
                st.write(f"White pixels: {white_eroded:,}")
                st.write(f"Change: {change_eroded:.1f}%")
            
            with col3:
                st.subheader("Dilation")
                st.image(dilated, use_column_width=True)
                white_dilated = np.sum(dilated == 255)
                change_dilated = ((white_dilated - white_orig) / white_orig) * 100
                st.write(f"White pixels: {white_dilated:,}")
                st.write(f"Change: {change_dilated:.1f}%")
            
            col4, col5 = st.columns(2)
            
            with col4:
                st.subheader("Opening")
                st.image(opened, use_column_width=True)
                white_opened = np.sum(opened == 255)
                change_opened = ((white_opened - white_orig) / white_orig) * 100
                st.write(f"White pixels: {white_opened:,}")
                st.write(f"Change: {change_opened:.1f}%")
            
            with col5:
                st.subheader("Closing")
                st.image(closed, use_column_width=True)
                white_closed = np.sum(closed == 255)
                change_closed = ((white_closed - white_orig) / white_orig) * 100
                st.write(f"White pixels: {white_closed:,}")
                st.write(f"Change: {change_closed:.1f}%")
        
        # Advanced morphological operations
        if st.checkbox("🔬 Advanced Morphological Operations"):
            st.subheader("Gradient and Top-Hat Transforms")
            
            kernel_adv = np.ones((5, 5), np.uint8)
            
            # Morphological gradient
            gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel_adv)
            
            # Top hat (difference between input and opening)
            tophat = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel_adv)
            
            # Black hat (difference between closing and input)
            blackhat = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel_adv)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Morphological Gradient**")
                st.image(gradient, use_column_width=True)
                st.write("Dilation - Erosion = Edge detection")
            
            with col2:
                st.write("**Top Hat**")
                st.image(tophat, use_column_width=True)
                st.write("Input - Opening = Small bright details")
            
            with col3:
                st.write("**Black Hat**")
                st.image(blackhat, use_column_width=True)
                st.write("Closing - Input = Small dark details")
        
        # Practical applications
        if st.checkbox("🎯 Practical Applications"):
            st.subheader("Real-world Use Cases")
            
            applications = {
                "Text Processing": {
                    "Operations": "Opening → Closing",
                    "Purpose": "Clean text, remove noise, fill character gaps",
                    "Example": "OCR preprocessing, document analysis"
                },
                "Medical Imaging": {
                    "Operations": "Closing → Opening",
                    "Purpose": "Fill tissue gaps, remove artifacts",
                    "Example": "Cell counting, tumor detection"
                },
                "Industrial Inspection": {
                    "Operations": "Erosion → Dilation",
                    "Purpose": "Detect defects, measure features",
                    "Example": "PCB inspection, quality control"
                },
                "Fingerprint Processing": {
                    "Operations": "Closing → Thinning",
                    "Purpose": "Connect ridges, extract minutiae",
                    "Example": "Biometric identification"
                }
            }
            
            for app, details in applications.items():
                with st.expander(app):
                    st.write(f"**Operations:** {details['Operations']}")
                    st.write(f"**Purpose:** {details['Purpose']}")
                    st.write(f"**Example:** {details['Example']}")
    
    else:
        st.info("👆 Please upload an image or click 'Use Default Image' to begin the demonstration.")
    
    # Key takeaways
    st.subheader("🎯 Key Takeaways")
    st.write("""
    - Morphological operations work on binary images using structuring elements
    - Erosion shrinks objects, dilation expands them
    - Opening removes noise while preserving large structures
    - Closing fills gaps and connects broken parts
    - Kernel size and shape determine the operation's effect
    """)
