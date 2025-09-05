import streamlit as st
import cv2
import numpy as np
from utils.image_utils import load_image, get_image_info

def show():
    st.header("المحاضرة الأولى: مقدمة في الصور الرقمية")
    
    # Theory section
    st.subheader("النظرية")
    st.write("""
    الصور الرقمية عبارة عن مصفوفات ثنائية الأبعاد من البكسلات، حيث يمثل كل بكسل قيم الألوان أو الكثافة.
    يمكن أن تكون الصور رمادية (قناة واحدة) أو ملونة (قنوات متعددة مثل RGB).
    فهم خصائص الصورة مثل الأبعاد والقنوات وأنواع البيانات أمر أساسي.
    معالجة الصور تتضمن التلاعب بقيم البكسل هذه لتحسين أو استخراج المعلومات.
    """)
    
    st.subheader("العرض التفاعلي")
    
    # Load image
    image = load_image()
    
    if image is not None:
        # Get image information
        info = get_image_info(image)
        
        # Display image and information
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("الصورة الأصلية")
            if len(image.shape) == 3:
                # Convert BGR to RGB for display
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.image(image_rgb, use_container_width=True)
            else:
                st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("معلومات الصورة")
            st.write(f"**الأبعاد:** {info['size']}")
            st.write(f"**العرض:** {info['width']} بكسل")
            st.write(f"**الارتفاع:** {info['height']} بكسل")
            st.write(f"**القنوات:** {info['channels']}")
            st.write(f"**إجمالي البكسلات:** {info['total_pixels']:,}")
            st.write(f"**نوع البيانات:** {info['data_type']}")
            
            # Color space information
            if info['channels'] == 3:
                st.write("**مساحة الألوان:** RGB (ملونة)")
            elif info['channels'] == 1:
                st.write("**مساحة الألوان:** رمادية")
            else:
                st.write(f"**مساحة الألوان:** {info['channels']} قنوات")
        
        # Additional analysis
        st.subheader("تحليل قيم البكسل")
        
        col3, col4 = st.columns(2)
        
        with col3:
            if len(image.shape) == 3:
                # Color image statistics
                st.write("**إحصائيات القنوات:**")
                for i, channel in enumerate(['الأزرق', 'الأخضر', 'الأحمر']):
                    mean_val = np.mean(image[:, :, i])
                    std_val = np.std(image[:, :, i])
                    st.write(f"{channel}: المتوسط = {mean_val:.1f}, الانحراف = {std_val:.1f}")
            else:
                # Grayscale statistics
                mean_val = np.mean(image)
                std_val = np.std(image)
                min_val = np.min(image)
                max_val = np.max(image)
                st.write("**إحصائيات الكثافة:**")
                st.write(f"المتوسط: {mean_val:.1f}")
                st.write(f"الانحراف المعياري: {std_val:.1f}")
                st.write(f"الحد الأدنى: {min_val}")
                st.write(f"الحد الأقصى: {max_val}")
        
        with col4:
            # Histogram
            st.write("**معاينة المدرج التكراري:**")
            if len(image.shape) == 3:
                # Convert to grayscale for histogram
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                hist, bins = np.histogram(gray, bins=50, range=(0, 256))
            else:
                hist, bins = np.histogram(image, bins=50, range=(0, 256))
            
            st.bar_chart(hist)
    
    else:
        st.info("يرجى رفع صورة أو النقر على 'استخدام الصورة الافتراضية' لبدء العرض التوضيحي.")
    
    # Key takeaways
    st.subheader("النقاط الرئيسية")
    st.write("""
    - الصور الرقمية عبارة عن مصفوفات من قيم البكسل
    - أبعاد الصورة تحدد الدقة وحجم الملف
    - القنوات تحدد تمثيل الألوان (1=رمادي، 3=ملون)
    - قيم البكسل عادة ما تتراوح من 0-255 للصور 8-بت
    - فهم خصائص الصورة ضروري للمعالجة
    """)
