import streamlit as st
import cv2
import numpy as np
from utils.image_utils import load_image, display_images
from utils.processing import rotate_image, scale_image, flip_image

def show():
    st.header("المحاضرة الثامنة: التحويلات الهندسية")
    
    # Theory section
    st.subheader("النظرية")
    st.write("""
    التحويلات الهندسية تعدل العلاقات المكانية بين البكسلات في الصورة.
    تحويلات الدوران تتضمن دوران الصور حول نقطة مركز باستخدام مصفوفات التحويل.
    التكبير يغير حجم الصورة مع الحفاظ على نسبة العرض أو السماح بالتكبير المستقل.
    القلب ينشئ صور مرآة على طول المحاور الأفقية أو العمودية أو كليهما معاً.
    """)
    
    st.subheader("العرض التفاعلي")
    
    # Load image
    image = load_image()
    
    if image is not None:
        # Transformation selection
        transformation = st.selectbox(
            "اختر التحويل الهندسي:",
            ["الدوران", "التكبير", "القلب", "القص", "التحويل المعقد"]
        )
        
        if transformation == "الدوران":
            st.subheader("دوران الصورة")
            
            col1, col2 = st.columns(2)
            
            with col1:
                angle = st.slider("زاوية الدوران (بالدرجات)", -180, 180, 0, 5)
                
            with col2:
                st.write("**اتجاه الدوران:**")
                if angle > 0:
                    st.write("↻ عكس عقارب الساعة")
                elif angle < 0:
                    st.write("↺ مع عقارب الساعة")
                else:
                    st.write("→ بدون دوران")
            
            # Apply rotation
            rotated = rotate_image(image, angle)
            
            # Display comparison
            display_images(image, rotated, ["الأصلية", f"مدورة {angle}°"])
            
            # Show rotation matrix
            if st.checkbox("Show Rotation Matrix"):
                height, width = image.shape[:2]
                center = (width // 2, height // 2)
                
                # Calculate rotation matrix
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                
                st.write("**2D Rotation Matrix:**")
                st.code(f"""
M = [[{rotation_matrix[0,0]:.3f}, {rotation_matrix[0,1]:.3f}, {rotation_matrix[0,2]:.3f}],
     [{rotation_matrix[1,0]:.3f}, {rotation_matrix[1,1]:.3f}, {rotation_matrix[1,2]:.3f}]]
                """)
                
                st.write("**Matrix Components:**")
                st.write(f"- Rotation: [[cos({angle}°), -sin({angle}°)], [sin({angle}°), cos({angle}°)]]")
                st.write(f"- Translation: [{rotation_matrix[0,2]:.1f}, {rotation_matrix[1,2]:.1f}]")
                st.write(f"- Center: {center}")
            
            # Show coordinate mapping
            if st.checkbox("Demonstrate Coordinate Mapping"):
                st.write("**Corner Coordinate Transformation:**")
                
                height, width = image.shape[:2]
                corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
                
                # Apply transformation to corners
                rotation_matrix = cv2.getRotationMatrix2D((width//2, height//2), angle, 1.0)
                
                transformed_corners = []
                for corner in corners:
                    # Add homogeneous coordinate
                    homogeneous = np.array([corner[0], corner[1], 1])
                    # Apply transformation
                    transformed = rotation_matrix @ homogeneous
                    transformed_corners.append(transformed)
                
                corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
                
                for i, (name, orig, trans) in enumerate(zip(corner_names, corners, transformed_corners)):
                    st.write(f"**{name}:** ({orig[0]:.0f}, {orig[1]:.0f}) → ({trans[0]:.1f}, {trans[1]:.1f})")
        
        elif transformation == "Scaling":
            st.subheader("📏 Image Scaling")
            
            scale_mode = st.radio(
                "Scaling Mode:",
                ["Uniform Scaling", "Non-uniform Scaling"]
            )
            
            if scale_mode == "Uniform Scaling":
                scale_factor = st.slider("Scale Factor", 0.1, 3.0, 1.0, 0.1)
                scaled = scale_image(image, scale_factor)
                
                # Display comparison
                display_images(image, scaled, ["Original", f"Scaled {scale_factor}x"])
                
                # Show size information
                orig_height, orig_width = image.shape[:2]
                new_height, new_width = scaled.shape[:2]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Original Size:**")
                    st.write(f"{orig_width} × {orig_height} pixels")
                    st.write(f"Total: {orig_width * orig_height:,} pixels")
                
                with col2:
                    st.write("**Scaled Size:**")
                    st.write(f"{new_width} × {new_height} pixels")
                    st.write(f"Total: {new_width * new_height:,} pixels")
                    
                    pixel_ratio = (new_width * new_height) / (orig_width * orig_height)
                    st.write(f"Pixel ratio: {pixel_ratio:.2f}x")
            
            else:  # Non-uniform scaling
                col1, col2 = st.columns(2)
                
                with col1:
                    width_scale = st.slider("Width Scale Factor", 0.1, 3.0, 1.0, 0.1)
                
                with col2:
                    height_scale = st.slider("Height Scale Factor", 0.1, 3.0, 1.0, 0.1)
                
                # Apply non-uniform scaling
                height, width = image.shape[:2]
                new_width = int(width * width_scale)
                new_height = int(height * height_scale)
                
                scaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                
                # Display comparison
                display_images(image, scaled, ["Original", f"Scaled W:{width_scale}x H:{height_scale}x"])
                
                # Aspect ratio analysis
                orig_aspect = width / height
                new_aspect = new_width / new_height
                
                st.write("**Aspect Ratio Analysis:**")
                st.write(f"Original: {orig_aspect:.3f} ({width}:{height})")
                st.write(f"Scaled: {new_aspect:.3f} ({new_width}:{new_height})")
                
                if abs(orig_aspect - new_aspect) > 0.01:
                    st.warning("⚠️ Aspect ratio changed - image may appear distorted")
                else:
                    st.success("✅ Aspect ratio preserved")
            
            # Interpolation methods
            if st.checkbox("Compare Interpolation Methods"):
                st.subheader("Interpolation Method Comparison")
                
                # Apply different interpolation methods
                scale_test = 2.0
                height, width = image.shape[:2]
                new_size = (int(width * scale_test), int(height * scale_test))
                
                nearest = cv2.resize(image, new_size, interpolation=cv2.INTER_NEAREST)
                linear = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
                cubic = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Nearest Neighbor**")
                    if len(nearest.shape) == 3:
                        nearest_rgb = cv2.cvtColor(nearest, cv2.COLOR_BGR2RGB)
                        st.image(nearest_rgb, use_column_width=True)
                    else:
                        st.image(nearest, use_column_width=True)
                    st.write("Fast, pixelated")
                
                with col2:
                    st.write("**Linear (Bilinear)**")
                    if len(linear.shape) == 3:
                        linear_rgb = cv2.cvtColor(linear, cv2.COLOR_BGR2RGB)
                        st.image(linear_rgb, use_column_width=True)
                    else:
                        st.image(linear, use_column_width=True)
                    st.write("Balanced quality/speed")
                
                with col3:
                    st.write("**Cubic (Bicubic)**")
                    if len(cubic.shape) == 3:
                        cubic_rgb = cv2.cvtColor(cubic, cv2.COLOR_BGR2RGB)
                        st.image(cubic_rgb, use_column_width=True)
                    else:
                        st.image(cubic, use_column_width=True)
                    st.write("Smooth, slower")
        
        elif transformation == "Flipping":
            st.subheader("🔄 Image Flipping")
            
            flip_type = st.selectbox(
                "Choose flip type:",
                ["Horizontal (Left ↔ Right)", "Vertical (Top ↔ Bottom)", "Both Axes"]
            )
            
            # Map flip types to OpenCV codes
            flip_codes = {
                "Horizontal (Left ↔ Right)": 1,
                "Vertical (Top ↔ Bottom)": 0,
                "Both Axes": -1
            }
            
            flip_code = flip_codes[flip_type]
            flipped = flip_image(image, flip_code)
            
            # Display comparison
            display_images(image, flipped, ["Original", f"Flipped ({flip_type})"])
            
            # Show flip explanation
            st.write("**Flip Operation:**")
            
            if flip_code == 1:
                st.write("- Horizontal flip: x' = width - x - 1")
                st.write("- Creates left-right mirror image")
                st.write("- Useful for data augmentation")
            elif flip_code == 0:
                st.write("- Vertical flip: y' = height - y - 1")
                st.write("- Creates top-bottom mirror image")
                st.write("- Less common in practice")
            else:
                st.write("- Both axes: x' = width - x - 1, y' = height - y - 1")
                st.write("- Equivalent to 180° rotation")
                st.write("- Creates point reflection")
            
            # Demonstrate with coordinates
            if st.checkbox("Show Coordinate Transformation"):
                height, width = image.shape[:2]
                
                # Sample points
                sample_points = [
                    (50, 50, "Top-Left Region"),
                    (width-50, 50, "Top-Right Region"),
                    (width//2, height//2, "Center"),
                    (50, height-50, "Bottom-Left Region"),
                    (width-50, height-50, "Bottom-Right Region")
                ]
                
                st.write("**Point Transformations:**")
                
                for x, y, description in sample_points:
                    if flip_code == 1:  # Horizontal
                        x_new, y_new = width - x - 1, y
                    elif flip_code == 0:  # Vertical
                        x_new, y_new = x, height - y - 1
                    else:  # Both
                        x_new, y_new = width - x - 1, height - y - 1
                    
                    st.write(f"**{description}:** ({x}, {y}) → ({x_new}, {y_new})")
        
        elif transformation == "Cropping":
            st.subheader("✂️ Image Cropping")
            
            height, width = image.shape[:2]
            
            st.write("**Define Crop Region:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                x_start = st.slider("X Start", 0, width-50, 0)
                y_start = st.slider("Y Start", 0, height-50, 0)
            
            with col2:
                x_end = st.slider("X End", x_start+50, width, width)
                y_end = st.slider("Y End", y_start+50, height, height)
            
            # Perform cropping
            cropped = image[y_start:y_end, x_start:x_end]
            
            # Display comparison
            display_images(image, cropped, ["Original", f"Cropped ({x_end-x_start}×{y_end-y_start})"])
            
            # Show crop information
            crop_width = x_end - x_start
            crop_height = y_end - y_start
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Original Image:**")
                st.write(f"Size: {width} × {height}")
                st.write(f"Total pixels: {width * height:,}")
            
            with col2:
                st.write("**Cropped Region:**")
                st.write(f"Size: {crop_width} × {crop_height}")
                st.write(f"Total pixels: {crop_width * crop_height:,}")
                
                retention = (crop_width * crop_height) / (width * height) * 100
                st.write(f"Pixel retention: {retention:.1f}%")
            
            # Region visualization
            if st.checkbox("Show Crop Region on Original"):
                # Create a copy with crop region highlighted
                highlighted = image.copy()
                
                # Draw rectangle around crop region
                cv2.rectangle(highlighted, (x_start, y_start), (x_end, y_end), (0, 255, 0), 3)
                
                st.subheader("Crop Region Preview")
                if len(highlighted.shape) == 3:
                    highlighted_rgb = cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB)
                    st.image(highlighted_rgb, use_column_width=True)
                else:
                    st.image(highlighted, use_column_width=True)
        
        elif transformation == "Complex Transform":
            st.subheader("🌟 Complex Transformation Pipeline")
            
            st.write("Apply multiple transformations in sequence:")
            
            # Initialize with original image
            current_image = image.copy()
            transformation_steps = []
            
            # Step 1: Rotation
            if st.checkbox("1. Apply Rotation"):
                angle = st.slider("Rotation angle", -45, 45, 0, 5)
                if angle != 0:
                    current_image = rotate_image(current_image, angle)
                    transformation_steps.append(f"Rotated {angle}°")
            
            # Step 2: Scaling
            if st.checkbox("2. Apply Scaling"):
                scale = st.slider("Scale factor", 0.5, 2.0, 1.0, 0.1)
                if scale != 1.0:
                    current_image = scale_image(current_image, scale)
                    transformation_steps.append(f"Scaled {scale}x")
            
            # Step 3: Flipping
            if st.checkbox("3. Apply Flipping"):
                flip_option = st.selectbox("Flip direction", ["None", "Horizontal", "Vertical", "Both"])
                
                flip_map = {"None": None, "Horizontal": 1, "Vertical": 0, "Both": -1}
                
                if flip_option != "None":
                    current_image = flip_image(current_image, flip_map[flip_option])
                    transformation_steps.append(f"Flipped {flip_option}")
            
            # Display pipeline result
            if transformation_steps:
                display_images(image, current_image, ["Original", "Transformed"])
                
                st.write("**Transformation Pipeline:**")
                for i, step in enumerate(transformation_steps, 1):
                    st.write(f"{i}. {step}")
                
                # Show size changes
                orig_h, orig_w = image.shape[:2]
                final_h, final_w = current_image.shape[:2]
                
                st.write(f"**Size Change:** {orig_w}×{orig_h} → {final_w}×{final_h}")
                
                # Option to save result
                from utils.image_utils import download_image
                download_image(current_image, "transformed_image.png")
            
            else:
                st.info("👆 Select transformations to see the pipeline in action")
        
        # Practical applications
        if st.checkbox("🎯 Real-world Applications"):
            st.subheader("Geometric Transformations in Practice")
            
            applications = {
                "Data Augmentation": {
                    "Transforms": "Rotation, Flipping, Scaling",
                    "Purpose": "Increase training data diversity",
                    "Use Case": "Machine learning, deep neural networks"
                },
                "Image Registration": {
                    "Transforms": "Rotation, Translation, Scaling",
                    "Purpose": "Align images from different sources",
                    "Use Case": "Medical imaging, satellite imagery"
                },
                "Document Processing": {
                    "Transforms": "Rotation, Perspective correction",
                    "Purpose": "Straighten scanned documents",
                    "Use Case": "OCR, document digitization"
                },
                "Computer Vision": {
                    "Transforms": "Affine, Perspective, Homography",
                    "Purpose": "Normalize object appearance",
                    "Use Case": "Object detection, face recognition"
                }
            }
            
            for app, details in applications.items():
                with st.expander(app):
                    st.write(f"**Transforms:** {details['Transforms']}")
                    st.write(f"**Purpose:** {details['Purpose']}")
                    st.write(f"**Use Case:** {details['Use Case']}")
    
    else:
        st.info("👆 Please upload an image or click 'Use Default Image' to begin the demonstration.")
    
    # Key takeaways
    st.subheader("🎯 Key Takeaways")
    st.write("""
    - Geometric transformations modify spatial pixel relationships
    - Rotation uses transformation matrices and interpolation
    - Scaling can be uniform or non-uniform with different interpolation methods
    - Flipping creates mirror images along specified axes
    - Complex pipelines combine multiple transformations sequentially
    """)
