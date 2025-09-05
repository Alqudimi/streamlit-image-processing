import streamlit as st
import cv2
import numpy as np
from utils.image_utils import load_image, display_images
from utils.processing import sobel_edge_detection, laplacian_edge_detection, canny_edge_detection

def show():
    st.header("المحاضرة السادسة: تقنيات كشف الحواف")
    
    # Theory section
    st.subheader("النظرية")
    st.write("""
    كشف الحواف يحدد الحدود بين المناطق ذات الكثافات أو الألوان المختلفة.
    مشغل سوبل يستخدم المشتقات الاتجاهية لإيجاد الحواف الأفقية والعمودية.
    كاشف لابلاسيان يجد الحواف باستخدام المشتقات من الدرجة الثانية، حساس للضوضاء.
    كشف الحواف كاني يجمع بين التنعيم الغاوسي وحساب التدرج والعتبة الهستيرية.
    """)
    
    st.subheader("العرض التفاعلي")
    
    # Load image
    image = load_image()
    
    if image is not None:
        # Convert to grayscale for edge detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Edge detection method selection
        method = st.selectbox(
            "اختر طريقة كشف الحواف:",
            ["سوبل", "لابلاسيان", "كاني", "مقارنة الكل"]
        )
        
        if method == "سوبل":
            st.subheader("كشف الحواف بطريقة سوبل")
            
            # Sobel parameters
            show_components = st.checkbox("عرض مكونات X و Y منفصلة")
            
            if show_components:
                # Calculate Sobel components
                sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("سوبل X (الحواف العمودية)")
                    sobel_x_display = np.uint8(np.clip(np.absolute(sobel_x), 0, 255))
                    st.image(sobel_x_display, use_container_width=True)
                    st.write("يكشف الحواف العمودية")
                
                with col2:
                    st.subheader("سوبل Y (الحواف الأفقية)")
                    sobel_y_display = np.uint8(np.clip(np.absolute(sobel_y), 0, 255))
                    st.image(sobel_y_display, use_container_width=True)
                    st.write("يكشف الحواف الأفقية")
                
                with col3:
                    st.subheader("المقدار المدمج")
                    sobel_combined_display = np.uint8(np.clip(sobel_combined, 0, 255))
                    st.image(sobel_combined_display, use_container_width=True)
                    st.write("√(Sx² + Sy²)")
                
                # Show original for comparison
                st.subheader("الصورة الأصلية")
                st.image(gray, use_container_width=True)
                
            else:
                # Standard Sobel edge detection
                edges = sobel_edge_detection(image)
                display_images(gray, edges, ["الأصلية", "حواف سوبل"])
            
            # Show Sobel kernels
            if st.checkbox("Show Sobel Kernels"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Sobel X Kernel:**")
                    sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                    st.code(str(sobel_x_kernel))
                
                with col2:
                    st.write("**Sobel Y Kernel:**")
                    sobel_y_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
                    st.code(str(sobel_y_kernel))
        
        elif method == "Laplacian":
            st.subheader("🌊 Laplacian Edge Detection")
            
            # Apply Laplacian
            edges = laplacian_edge_detection(image)
            
            display_images(gray, edges, ["Original", "Laplacian Edges"])
            
            # Show Laplacian kernel
            if st.checkbox("Show Laplacian Kernel"):
                st.write("**Standard Laplacian Kernel:**")
                laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
                st.code(str(laplacian_kernel))
                
                st.write("**Alternative Laplacian Kernel:**")
                laplacian_kernel_alt = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
                st.code(str(laplacian_kernel_alt))
            
            # Noise sensitivity demonstration
            if st.checkbox("Demonstrate Noise Sensitivity"):
                from utils.processing import add_noise
                
                noise_level = st.slider("Noise Level", 0.0, 0.1, 0.05, 0.01)
                noisy_image = add_noise(image, "gaussian", noise_level)
                
                if len(noisy_image.shape) == 3:
                    noisy_gray = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY)
                else:
                    noisy_gray = noisy_image
                
                noisy_edges = laplacian_edge_detection(noisy_image)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Clean Image Edges")
                    st.image(edges, use_container_width=True)
                
                with col2:
                    st.subheader("Noisy Image Edges")
                    st.image(noisy_edges, use_container_width=True)
                
                st.warning("⚠️ Laplacian is sensitive to noise due to second-order derivatives")
        
        elif method == "Canny":
            st.subheader("🎯 Canny Edge Detection")
            
            col1, col2 = st.columns(2)
            
            with col1:
                low_threshold = st.slider("Low Threshold", 0, 255, 50, 5)
            
            with col2:
                high_threshold = st.slider("High Threshold", 0, 255, 150, 5)
            
            if high_threshold <= low_threshold:
                st.warning("High threshold should be greater than low threshold")
                high_threshold = low_threshold + 50
            
            # Apply Canny edge detection
            edges = canny_edge_detection(image, low_threshold, high_threshold)
            
            display_images(gray, edges, ["Original", f"Canny Edges (L:{low_threshold}, H:{high_threshold})"])
            
            # Explain Canny process
            if st.checkbox("Show Canny Process Steps"):
                st.write("**Canny Edge Detection Steps:**")
                
                # Step 1: Gaussian blur
                blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
                
                # Step 2: Gradient calculation
                sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.write("**1. Gaussian Blur**")
                    st.image(blurred, use_container_width=True)
                
                with col2:
                    st.write("**2. Gradient Magnitude**")
                    gradient_display = np.uint8(np.clip(gradient_magnitude, 0, 255))
                    st.image(gradient_display, use_container_width=True)
                
                with col3:
                    st.write("**3. Non-Max Suppression**")
                    st.write("(Simplified visualization)")
                    st.image(gradient_display, use_container_width=True)
                
                with col4:
                    st.write("**4. Hysteresis**")
                    st.image(edges, use_container_width=True)
                
                st.write("""
                **Process Details:**
                1. **Gaussian Blur**: Reduces noise before edge detection
                2. **Gradient Calculation**: Finds edge strength and direction
                3. **Non-Maximum Suppression**: Thins edges to single pixels
                4. **Hysteresis Thresholding**: Uses two thresholds to link edges
                """)
            
            # Threshold analysis
            if st.checkbox("Threshold Impact Analysis"):
                # Show different threshold combinations
                thresholds = [(25, 75), (50, 150), (100, 200)]
                
                cols = st.columns(3)
                
                for i, (low, high) in enumerate(thresholds):
                    with cols[i]:
                        test_edges = canny_edge_detection(image, low, high)
                        st.write(f"**L:{low}, H:{high}**")
                        st.image(test_edges, use_container_width=True)
                        
                        # Count edge pixels
                        edge_pixels = np.sum(test_edges > 0)
                        total_pixels = test_edges.size
                        edge_percentage = (edge_pixels / total_pixels) * 100
                        st.write(f"Edge pixels: {edge_percentage:.1f}%")
        
        elif method == "Compare All":
            st.subheader("🔍 Edge Detection Comparison")
            
            # Apply all methods
            sobel_edges = sobel_edge_detection(image)
            laplacian_edges = laplacian_edge_detection(image)
            canny_edges = canny_edge_detection(image, 50, 150)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.subheader("Original")
                st.image(gray, use_container_width=True)
                st.write("Input grayscale image")
            
            with col2:
                st.subheader("Sobel")
                st.image(sobel_edges, use_container_width=True)
                st.write("Gradient-based, directional")
            
            with col3:
                st.subheader("Laplacian")
                st.image(laplacian_edges, use_container_width=True)
                st.write("Second derivative, noise sensitive")
            
            with col4:
                st.subheader("Canny")
                st.image(canny_edges, use_container_width=True)
                st.write("Multi-stage, optimal")
            
            # Quantitative comparison
            st.subheader("📊 Quantitative Comparison")
            
            def count_edge_pixels(edge_image):
                return np.sum(edge_image > 0)
            
            sobel_count = count_edge_pixels(sobel_edges)
            laplacian_count = count_edge_pixels(laplacian_edges)
            canny_count = count_edge_pixels(canny_edges)
            
            total_pixels = gray.size
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sobel_pct = (sobel_count / total_pixels) * 100
                st.metric("Sobel Edge Pixels", f"{sobel_pct:.1f}%")
            
            with col2:
                laplacian_pct = (laplacian_count / total_pixels) * 100
                st.metric("Laplacian Edge Pixels", f"{laplacian_pct:.1f}%")
            
            with col3:
                canny_pct = (canny_count / total_pixels) * 100
                st.metric("Canny Edge Pixels", f"{canny_pct:.1f}%")
            
            # Method characteristics
            st.subheader("📋 Method Characteristics")
            
            characteristics = {
                "Sobel": {
                    "Pros": "Fast, directional information, robust",
                    "Cons": "Thick edges, sensitive to noise",
                    "Best for": "Real-time applications, gradient analysis"
                },
                "Laplacian": {
                    "Pros": "Rotation invariant, single operator",
                    "Cons": "Very noise sensitive, no direction info",
                    "Best for": "Clean images, blob detection"
                },
                "Canny": {
                    "Pros": "Optimal detection, thin edges, less noise",
                    "Cons": "Slower, requires parameter tuning",
                    "Best for": "Precise edge maps, object detection"
                }
            }
            
            for method_name, chars in characteristics.items():
                with st.expander(f"{method_name} Edge Detection"):
                    st.write(f"**Pros:** {chars['Pros']}")
                    st.write(f"**Cons:** {chars['Cons']}")
                    st.write(f"**Best for:** {chars['Best for']}")
    
    else:
        st.info("👆 Please upload an image or click 'Use Default Image' to begin the demonstration.")
    
    # Key takeaways
    st.subheader("🎯 Key Takeaways")
    st.write("""
    - Edge detection identifies intensity boundaries in images
    - Sobel provides directional edge information using first derivatives
    - Laplacian uses second derivatives but is noise-sensitive
    - Canny is optimal but requires careful threshold selection
    - Choice depends on application requirements and image characteristics
    """)
