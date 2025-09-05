import streamlit as st
import cv2
import numpy as np
from utils.image_utils import load_image, display_images
from utils.processing import adjust_brightness_contrast, apply_negative, apply_threshold

def show():
    st.header("المحاضرة الثالثة: العمليات على مستوى البكسل")
    
    # Theory section
    st.subheader("النظرية")
    st.write("""
    عمليات البكسل تحول قيم البكسل الفردية باستخدام دوال رياضية.
    تعديل السطوع يضيف/يطرح قيم ثابتة لجميع البكسلات خطياً.
    تعديل التباين يضرب قيم البكسل لتعزيز أو تقليل الاختلافات.
    العتبة تحول الصور إلى ثنائية بمقارنة البكسلات بقيمة عتبة.
    """)
    
    st.subheader("العرض التفاعلي")
    
    # Load image
    image = load_image()
    
    if image is not None:
        # Operation selection
        operation = st.selectbox(
            "اختر عملية البكسل:",
            ["السطوع والتباين", "التحويل السلبي", "العتبة"]
        )
        
        if operation == "السطوع والتباين":
            st.subheader("تعديل السطوع والتباين")
            
            col1, col2 = st.columns(2)
            
            with col1:
                brightness = st.slider("السطوع", -100, 100, 0, 1)
                st.write("المعادلة: بكسل جديد = بكسل قديم + السطوع")
            
            with col2:
                contrast = st.slider("التباين", 0.1, 3.0, 1.0, 0.1)
                st.write("المعادلة: بكسل جديد = بكسل قديم × التباين")
            
            # Apply brightness and contrast
            adjusted = adjust_brightness_contrast(image, brightness, contrast)
            
            # Display comparison
            display_images(image, adjusted, ["الأصلية", "معدلة"])
            
            # Show histogram comparison
            if st.checkbox("Show Histogram Comparison"):
                col1, col2 = st.columns(2)
                
                with col1:
                    if len(image.shape) == 3:
                        gray_orig = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    else:
                        gray_orig = image
                    hist_orig, _ = np.histogram(gray_orig, bins=50, range=(0, 256))
                    st.bar_chart(hist_orig)
                    st.caption("Original Histogram")
                
                with col2:
                    if len(adjusted.shape) == 3:
                        gray_adj = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
                    else:
                        gray_adj = adjusted
                    hist_adj, _ = np.histogram(gray_adj, bins=50, range=(0, 256))
                    st.bar_chart(hist_adj)
                    st.caption("Adjusted Histogram")
        
        elif operation == "Negative Transform":
            st.subheader("🔄 Negative Transform")
            st.write("Formula: new_pixel = 255 - old_pixel")
            
            # Apply negative transform
            negative = apply_negative(image)
            
            # Display comparison
            display_images(image, negative, ["Original", "Negative"])
            
            # Show effect explanation
            st.write("**Effect:** Dark areas become light, light areas become dark")
            st.write("**Use cases:** Medical imaging, photography effects, preprocessing")
        
        elif operation == "Thresholding":
            st.subheader("📊 Thresholding Operations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                threshold_value = st.slider("Threshold Value", 0, 255, 127)
            
            with col2:
                threshold_type = st.selectbox(
                    "Threshold Type",
                    ["binary", "binary_inv", "truncate", "to_zero", "to_zero_inv"]
                )
            
            # Apply thresholding
            thresholded = apply_threshold(image, threshold_value, threshold_type)
            
            # Display comparison
            if len(image.shape) == 3:
                gray_input = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                display_images(gray_input, thresholded, ["Grayscale Input", f"Thresholded ({threshold_type})"])
            else:
                display_images(image, thresholded, ["Original", f"Thresholded ({threshold_type})"])
            
            # Explain threshold types
            st.write("**Threshold Types:**")
            explanations = {
                "binary": "pixel = 255 if pixel > threshold else 0",
                "binary_inv": "pixel = 0 if pixel > threshold else 255",
                "truncate": "pixel = threshold if pixel > threshold else pixel",
                "to_zero": "pixel = pixel if pixel > threshold else 0",
                "to_zero_inv": "pixel = 0 if pixel > threshold else pixel"
            }
            st.code(explanations[threshold_type])
        
        # Additional analysis
        st.subheader("📈 Pixel Statistics")
        
        if operation == "Brightness & Contrast":
            processed = adjusted
        elif operation == "Negative Transform":
            processed = negative
        elif operation == "Thresholding":
            processed = thresholded
        else:
            processed = image
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Original Mean", f"{np.mean(image):.1f}")
            st.metric("Original Std", f"{np.std(image):.1f}")
        
        with col2:
            st.metric("Processed Mean", f"{np.mean(processed):.1f}")
            st.metric("Processed Std", f"{np.std(processed):.1f}")
        
        with col3:
            st.metric("Min Value", f"{np.min(processed)}")
            st.metric("Max Value", f"{np.max(processed)}")
        
        # Interactive pixel inspection
        if st.checkbox("🔍 Pixel Inspector"):
            st.write("Click coordinates to inspect pixel values:")
            
            col1, col2 = st.columns(2)
            with col1:
                x_coord = st.number_input("X coordinate", 0, image.shape[1]-1, image.shape[1]//2)
            with col2:
                y_coord = st.number_input("Y coordinate", 0, image.shape[0]-1, image.shape[0]//2)
            
            if len(image.shape) == 3:
                orig_pixel = image[y_coord, x_coord]
                proc_pixel = processed[y_coord, x_coord] if len(processed.shape) == 3 else processed[y_coord, x_coord]
                st.write(f"Original RGB: {orig_pixel}")
                st.write(f"Processed: {proc_pixel}")
            else:
                orig_pixel = image[y_coord, x_coord]
                proc_pixel = processed[y_coord, x_coord]
                st.write(f"Original: {orig_pixel}")
                st.write(f"Processed: {proc_pixel}")
    
    else:
        st.info("👆 Please upload an image or click 'Use Default Image' to begin the demonstration.")
    
    # Key takeaways
    st.subheader("🎯 Key Takeaways")
    st.write("""
    - Pixel operations transform individual pixel values
    - Brightness shifts the entire intensity range
    - Contrast stretches or compresses the intensity range
    - Thresholding creates binary images for segmentation
    - These operations form the foundation of image enhancement
    """)
