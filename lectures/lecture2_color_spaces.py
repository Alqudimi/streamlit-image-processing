import streamlit as st
import cv2
import numpy as np
from utils.image_utils import load_image, display_images

def show():
    st.header("المحاضرة الثانية: مساحات الألوان وعمليات القنوات")
    
    # Theory section
    st.subheader("النظرية")
    st.write("""
    مساحات الألوان هي نماذج رياضية لتمثيل الألوان. RGB شائع لأجهزة العرض.
    HSV (درجة اللون، التشبع، القيمة) بديهي للمعالجة والتحليل القائم على الألوان.
    الدرجات الرمادية تقلل التعقيد وضرورية للعديد من خوارزميات الرؤية الحاسوبية.
    تقسيم القنوات يسمح بتحليل مكونات الألوان الفردية في الصورة.
    """)
    
    st.subheader("العرض التفاعلي")
    
    # Load image
    image = load_image()
    
    if image is not None:
        # Ensure image is in color
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Color space conversion options
        conversion_type = st.selectbox(
            "اختر تحويل مساحة الألوان:",
            ["RGB إلى رمادي", "RGB إلى HSV", "تقسيم القنوات"]
        )
        
        if conversion_type == "RGB إلى رمادي":
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Display comparison
            display_images(image, gray, ["الأصلية (ملونة)", "رمادية"])
            
            st.write("**العملية:** رمادي = 0.299×أحمر + 0.587×أخضر + 0.114×أزرق")
            
        elif conversion_type == "RGB إلى HSV":
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Display comparison
            display_images(image, hsv, ["الأصلية (RGB)", "تمثيل HSV"])
            
            st.write("**مكونات HSV:**")
            st.write("- **درجة اللون (Hue):** نوع اللون (0-179 في OpenCV)")
            st.write("- **التشبع (Saturation):** كثافة اللون (0-255)")
            st.write("- **القيمة (Value):** السطوع (0-255)")
            
        elif conversion_type == "تقسيم القنوات":
            # Split RGB channels
            b, g, r = cv2.split(image)
            
            # Create visualizations for each channel
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("القناة الزرقاء")
                st.image(b, use_container_width=True, clamp=True)
                st.write(f"المتوسط: {np.mean(b):.1f}")
            
            with col2:
                st.subheader("القناة الخضراء")
                st.image(g, use_container_width=True, clamp=True)
                st.write(f"المتوسط: {np.mean(g):.1f}")
            
            with col3:
                st.subheader("القناة الحمراء")
                st.image(r, use_container_width=True, clamp=True)
                st.write(f"المتوسط: {np.mean(r):.1f}")
            
            # Original image
            st.subheader("الصورة الأصلية")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, use_container_width=True)
        
        # Additional controls
        st.subheader("خيارات متقدمة")
        
        if st.checkbox("عرض تحليل قنوات HSV"):
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**قناة درجة اللون**")
                st.image(h, use_container_width=True, clamp=True)
                st.write(f"الألوان السائدة: {np.unique(h)[:5]}")
            
            with col2:
                st.write("**قناة التشبع**")
                st.image(s, use_container_width=True, clamp=True)
                st.write(f"متوسط التشبع: {np.mean(s):.1f}")
            
            with col3:
                st.write("**قناة القيمة**")
                st.image(v, use_container_width=True, clamp=True)
                st.write(f"متوسط السطوع: {np.mean(v):.1f}")
        
        # Color manipulation
        if st.checkbox("معالجة الألوان التفاعلية"):
            st.write("تعديل قنوات الألوان الفردية:")
            
            # Channel multipliers
            col1, col2, col3 = st.columns(3)
            
            with col1:
                b_mult = st.slider("مضاعف الأزرق", 0.0, 2.0, 1.0, 0.1)
            with col2:
                g_mult = st.slider("مضاعف الأخضر", 0.0, 2.0, 1.0, 0.1)
            with col3:
                r_mult = st.slider("مضاعف الأحمر", 0.0, 2.0, 1.0, 0.1)
            
            # Apply multipliers
            modified = image.copy().astype(np.float32)
            modified[:, :, 0] *= b_mult  # Blue
            modified[:, :, 1] *= g_mult  # Green
            modified[:, :, 2] *= r_mult  # Red
            modified = np.clip(modified, 0, 255).astype(np.uint8)
            
            # Display result
            display_images(image, modified, ["الأصلية", "معدلة الألوان"])
    
    else:
        st.info("يرجى رفع صورة أو النقر على 'استخدام الصورة الافتراضية' لبدء العرض التوضيحي.")
    
    # Key takeaways
    st.subheader("النقاط الرئيسية")
    st.write("""
    - RGB هو نموذج ألوان إضافي يستخدم في الشاشات
    - HSV يفصل معلومات الألوان عن الكثافة
    - التحويل الرمادي يقلل البيانات مع الحفاظ على البنية
    - عمليات القنوات تمكن من معالجة الألوان المستهدفة
    - مساحات الألوان المختلفة تناسب مهام معالجة مختلفة
    """)
