import streamlit as st
import cv2
import numpy as np
from PIL import Image
import base64
import io

def load_default_image():
    """Create a default sample image for demonstrations."""
    # Create a simple colorful image with geometric shapes
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Background gradient
    for i in range(300):
        img[i, :] = [100 + i//3, 150, 200 - i//4]
    
    # Add some shapes
    cv2.rectangle(img, (50, 50), (150, 150), (255, 100, 100), -1)
    cv2.circle(img, (300, 100), 50, (100, 255, 100), -1)
    cv2.ellipse(img, (200, 200), (80, 40), 45, 0, 360, (100, 100, 255), -1)
    
    # Add some text
    cv2.putText(img, 'Sample', (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return img

def load_image():
    """Load image from file upload or use default."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "اختر ملف صورة", 
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff']
        )
    
    with col2:
        use_default = st.button("استخدام الصورة الافتراضية")
    
    if uploaded_file is not None:
        # Read uploaded image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return img_array
    
    elif use_default or 'default_image' not in st.session_state:
        # Use default image
        img_array = load_default_image()
        st.session_state['default_image'] = img_array
        return img_array
    
    elif 'default_image' in st.session_state:
        return st.session_state['default_image']
    
    return None

def display_images(original, processed, titles=["الأصلية", "معالجة"]):
    """Display original and processed images side by side."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(titles[0])
        if len(original.shape) == 3:
            # Convert BGR to RGB for display
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            st.image(original_rgb, use_column_width=True)
        else:
            st.image(original, use_column_width=True, clamp=True)
    
    with col2:
        st.subheader(titles[1])
        if len(processed.shape) == 3:
            # Convert BGR to RGB for display
            processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            st.image(processed_rgb, use_column_width=True)
        else:
            st.image(processed, use_column_width=True, clamp=True)

def download_image(image, filename="processed_image.png"):
    """Create download button for processed image."""
    if len(image.shape) == 3:
        # Convert BGR to RGB for saving
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
    else:
        pil_image = Image.fromarray(image)
    
    # Convert to bytes
    buf = io.BytesIO()
    pil_image.save(buf, format='PNG')
    byte_im = buf.getvalue()
    
    st.download_button(
        label="تحميل الصورة المعالجة",
        data=byte_im,
        file_name=filename,
        mime="image/png"
    )

def get_image_info(image):
    """Get basic information about the image."""
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    
    return {
        'width': width,
        'height': height,
        'channels': channels,
        'size': f"{width} x {height}",
        'total_pixels': width * height,
        'data_type': str(image.dtype)
    }
