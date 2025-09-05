import streamlit as st
import cv2
import numpy as np
from utils.image_utils import load_image, display_images
from utils.processing import apply_gaussian_blur, apply_median_blur, apply_custom_kernel

def show():
    st.header("المحاضرة الرابعة: ترشيح الصور والالتفاف")
    
    # Theory section
    st.subheader("النظرية")
    st.write("""
    ترشيح الصور يطبق عمليات رياضية باستخدام النوى (مصفوفات صغيرة) لتحسين أو تعديل الصور.
    الالتفاف ينزلق بالنواة عبر الصورة، محسوباً المجاميع المرجحة في كل موضع.
    التمويه الغاوسي يقلل الضوضاء مع الحفاظ على الحواف، التمويه الوسطي يزيل ضوضاء الملح والفلفل.
    النوى المخصصة يمكنها شحذ الصور، كشف الحواف، أو إنشاء تأثيرات فنية.
    """)
    
    st.subheader("العرض التفاعلي")
    
    # Load image
    image = load_image()
    
    if image is not None:
        # Filter selection
        filter_type = st.selectbox(
            "اختر عملية الترشيح:",
            ["التمويه الغاوسي", "التمويه الوسطي", "النوى المخصصة", "المرشح الثنائي"]
        )
        
        if filter_type == "Gaussian Blur":
            st.subheader("🌫️ Gaussian Blur")
            
            kernel_size = st.slider("Kernel Size", 1, 31, 5, 2)
            st.write(f"Using {kernel_size}×{kernel_size} Gaussian kernel")
            
            # Apply Gaussian blur
            blurred = apply_gaussian_blur(image, kernel_size)
            
            # Display comparison
            display_images(image, blurred, ["Original", f"Gaussian Blur (k={kernel_size})"])
            
            # Show kernel visualization
            if st.checkbox("Show Gaussian Kernel"):
                # Create Gaussian kernel for visualization
                sigma = kernel_size / 6.0
                kernel_1d = cv2.getGaussianKernel(kernel_size, sigma)
                kernel_2d = kernel_1d @ kernel_1d.T
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Gaussian Kernel:**")
                    st.image(kernel_2d, width=200)
                with col2:
                    st.write("**Properties:**")
                    st.write(f"Size: {kernel_size}×{kernel_size}")
                    st.write(f"Sigma: {sigma:.2f}")
                    st.write(f"Sum: {np.sum(kernel_2d):.3f}")
        
        elif filter_type == "Median Blur":
            st.subheader("📊 Median Blur")
            
            kernel_size = st.slider("Kernel Size", 3, 15, 5, 2)
            st.write("Median blur replaces each pixel with the median value in its neighborhood")
            
            # Apply median blur
            median_blurred = apply_median_blur(image, kernel_size)
            
            # Display comparison
            display_images(image, median_blurred, ["Original", f"Median Blur (k={kernel_size})"])
            
            # Compare with Gaussian
            if st.checkbox("Compare with Gaussian Blur"):
                gaussian_blurred = apply_gaussian_blur(image, kernel_size)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Median Blur")
                    if len(median_blurred.shape) == 3:
                        median_rgb = cv2.cvtColor(median_blurred, cv2.COLOR_BGR2RGB)
                        st.image(median_rgb, use_column_width=True)
                    else:
                        st.image(median_blurred, use_column_width=True)
                
                with col2:
                    st.subheader("Gaussian Blur")
                    if len(gaussian_blurred.shape) == 3:
                        gaussian_rgb = cv2.cvtColor(gaussian_blurred, cv2.COLOR_BGR2RGB)
                        st.image(gaussian_rgb, use_column_width=True)
                    else:
                        st.image(gaussian_blurred, use_column_width=True)
                
                st.write("**Median** preserves edges better, **Gaussian** is smoother")
        
        elif filter_type == "Custom Kernels":
            st.subheader("🎨 Custom Kernel Filtering")
            
            kernel_preset = st.selectbox(
                "Choose kernel preset:",
                ["Sharpen", "Edge Detection", "Emboss", "Custom"]
            )
            
            # Define preset kernels
            kernels = {
                "Sharpen": np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]),
                "Edge Detection": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
                "Emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
            }
            
            if kernel_preset != "Custom":
                kernel = kernels[kernel_preset]
                st.write(f"**{kernel_preset} Kernel:**")
                st.code(str(kernel))
            else:
                st.write("**Custom 3×3 Kernel:**")
                col1, col2, col3 = st.columns(3)
                
                # Create custom kernel input
                kernel = np.zeros((3, 3))
                for i in range(3):
                    for j in range(3):
                        with [col1, col2, col3][j]:
                            if i == 0:
                                st.write(f"Row {i+1}")
                            kernel[i, j] = st.number_input(f"", key=f"k_{i}_{j}", value=0.0, step=0.1, format="%.1f")
            
            # Apply custom kernel
            filtered = apply_custom_kernel(image, kernel)
            
            # Display comparison
            display_images(image, filtered, ["Original", f"{kernel_preset} Filter"])
            
            # Show kernel properties
            st.write(f"**Kernel Sum:** {np.sum(kernel):.2f}")
            if np.sum(kernel) == 0:
                st.info("Zero-sum kernels detect features (edges, textures)")
            elif np.sum(kernel) == 1:
                st.info("Unit-sum kernels preserve image brightness")
            else:
                st.warning("Non-unit sum will change image brightness")
        
        elif filter_type == "Bilateral Filter":
            st.subheader("🎭 Bilateral Filter")
            st.write("Preserves edges while smoothing uniform regions")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                d = st.slider("Diameter", 5, 25, 9, 2)
            with col2:
                sigma_color = st.slider("Sigma Color", 10, 150, 75, 5)
            with col3:
                sigma_space = st.slider("Sigma Space", 10, 150, 75, 5)
            
            # Apply bilateral filter
            if len(image.shape) == 3:
                bilateral = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
            else:
                # Convert to 3-channel for bilateral filter
                image_3ch = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                bilateral_3ch = cv2.bilateralFilter(image_3ch, d, sigma_color, sigma_space)
                bilateral = cv2.cvtColor(bilateral_3ch, cv2.COLOR_BGR2GRAY)
            
            # Display comparison
            display_images(image, bilateral, ["Original", "Bilateral Filtered"])
            
            st.write("**Parameters:**")
            st.write(f"- **Diameter:** {d} (neighborhood size)")
            st.write(f"- **Sigma Color:** {sigma_color} (color difference sensitivity)")
            st.write(f"- **Sigma Space:** {sigma_space} (spatial distance sensitivity)")
        
        # Filter comparison
        if st.checkbox("🔍 Filter Comparison"):
            st.subheader("Side-by-Side Filter Comparison")
            
            # Apply multiple filters
            gaussian = apply_gaussian_blur(image, 5)
            median = apply_median_blur(image, 5)
            
            sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = apply_custom_kernel(image, sharpen_kernel)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.write("**Original**")
                if len(image.shape) == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    st.image(image_rgb, use_column_width=True)
                else:
                    st.image(image, use_column_width=True)
            
            with col2:
                st.write("**Gaussian Blur**")
                if len(gaussian.shape) == 3:
                    gaussian_rgb = cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB)
                    st.image(gaussian_rgb, use_column_width=True)
                else:
                    st.image(gaussian, use_column_width=True)
            
            with col3:
                st.write("**Median Blur**")
                if len(median.shape) == 3:
                    median_rgb = cv2.cvtColor(median, cv2.COLOR_BGR2RGB)
                    st.image(median_rgb, use_column_width=True)
                else:
                    st.image(median, use_column_width=True)
            
            with col4:
                st.write("**Sharpened**")
                if len(sharpened.shape) == 3:
                    sharpened_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
                    st.image(sharpened_rgb, use_column_width=True)
                else:
                    st.image(sharpened, use_column_width=True)
    
    else:
        st.info("👆 Please upload an image or click 'Use Default Image' to begin the demonstration.")
    
    # Key takeaways
    st.subheader("🎯 Key Takeaways")
    st.write("""
    - Filtering uses convolution with kernels to modify images
    - Gaussian blur reduces noise with smooth transitions
    - Median blur removes impulse noise while preserving edges
    - Custom kernels enable sharpening, edge detection, and artistic effects
    - Bilateral filtering combines spatial and intensity information
    """)
