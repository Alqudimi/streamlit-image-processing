import streamlit as st
import cv2
import numpy as np
from utils.image_utils import load_image, display_images, download_image
from utils.processing import (
    adjust_brightness_contrast, apply_negative, apply_threshold,
    apply_gaussian_blur, apply_median_blur, apply_bilateral_filter,
    add_noise, sobel_edge_detection, laplacian_edge_detection, 
    canny_edge_detection, morphological_operation,
    rotate_image, scale_image, flip_image
)

def show():
    st.header("المحاضرة التاسعة: المشروع النهائي - خط معالجة الصور")
    
    # Theory section
    st.subheader("النظرية")
    st.write("""
    خطوط معالجة الصور تجمع عدة عمليات لتحقيق تحويلات معقدة.
    المعالجة المتسلسلة تسمح ببناء سير عمل متطور من العمليات الأساسية.
    تصميم الخط يتطلب فهم ترتيب العمليات وتحسين المعاملات.
    التطبيقات الحقيقية غالباً ما تتضمن عدة خطوات معالجة للحصول على النتائج المثلى.
    """)
    
    st.subheader("العرض التفاعلي - ابن خطك")
    
    # Load image
    image = load_image()
    
    if image is not None:
        # Initialize session state for pipeline
        if 'pipeline_steps' not in st.session_state:
            st.session_state.pipeline_steps = []
        if 'pipeline_image' not in st.session_state:
            st.session_state.pipeline_image = image.copy()
        
        # Reset pipeline button
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("إعادة تعيين الخط"):
                st.session_state.pipeline_steps = []
                st.session_state.pipeline_image = image.copy()
                st.rerun()
        
        with col2:
            if st.button("تراجع عن الخطوة الأخيرة"):
                if st.session_state.pipeline_steps:
                    st.session_state.pipeline_steps.pop()
                    # Rebuild pipeline from scratch
                    st.session_state.pipeline_image = image.copy()
                    for step in st.session_state.pipeline_steps:
                        st.session_state.pipeline_image = apply_pipeline_step(
                            st.session_state.pipeline_image, step
                        )
                    st.rerun()
        
        with col3:
            st.write(f"**خطوات الخط:** {len(st.session_state.pipeline_steps)}")
        
        # Pipeline builder
        st.subheader("بناء الخط")
        
        # Operation categories
        operation_category = st.selectbox(
            "اختر فئة العملية:",
            ["العمليات الأساسية", "الترشيح", "كشف الحواف", "المورفولوجيا", "الهندسية", "الضوضاء وإزالتها"]
        )
        
        # Operation selection based on category
        operation = None
        params = {}
        
        if operation_category == "Basic Operations":
            operation = st.selectbox(
                "Select operation:",
                ["Brightness/Contrast", "Negative", "Threshold", "Color Space"]
            )
            
            if operation == "Brightness/Contrast":
                col1, col2 = st.columns(2)
                with col1:
                    brightness = st.slider("Brightness", -100, 100, 0, key="pipeline_brightness")
                with col2:
                    contrast = st.slider("Contrast", 0.1, 3.0, 1.0, 0.1, key="pipeline_contrast")
                
                params = {"brightness": brightness, "contrast": contrast}
                
            elif operation == "Negative":
                st.write("Apply negative transformation")
                params = {}
                
            elif operation == "Threshold":
                col1, col2 = st.columns(2)
                with col1:
                    thresh_value = st.slider("Threshold Value", 0, 255, 127, key="pipeline_thresh")
                with col2:
                    thresh_type = st.selectbox("Threshold Type", 
                                             ["binary", "binary_inv", "truncate", "to_zero", "to_zero_inv"],
                                             key="pipeline_thresh_type")
                
                params = {"threshold_value": thresh_value, "threshold_type": thresh_type}
                
            elif operation == "Color Space":
                conversion = st.selectbox("Conversion", ["To Grayscale", "To HSV"], key="pipeline_color")
                params = {"conversion": conversion}
        
        elif operation_category == "Filtering":
            operation = st.selectbox(
                "Select filter:",
                ["Gaussian Blur", "Median Blur", "Bilateral Filter"]
            )
            
            if operation == "Gaussian Blur":
                kernel_size = st.slider("Kernel Size", 3, 31, 5, 2, key="pipeline_gauss_kernel")
                params = {"kernel_size": kernel_size}
                
            elif operation == "Median Blur":
                kernel_size = st.slider("Kernel Size", 3, 15, 5, 2, key="pipeline_median_kernel")
                params = {"kernel_size": kernel_size}
                
            elif operation == "Bilateral Filter":
                col1, col2, col3 = st.columns(3)
                with col1:
                    d = st.slider("Diameter", 5, 25, 9, 2, key="pipeline_bilateral_d")
                with col2:
                    sigma_color = st.slider("Sigma Color", 10, 150, 75, 5, key="pipeline_bilateral_sc")
                with col3:
                    sigma_space = st.slider("Sigma Space", 10, 150, 75, 5, key="pipeline_bilateral_ss")
                
                params = {"d": d, "sigma_color": sigma_color, "sigma_space": sigma_space}
        
        elif operation_category == "Edge Detection":
            operation = st.selectbox(
                "Select edge detector:",
                ["Sobel", "Laplacian", "Canny"]
            )
            
            if operation == "Canny":
                col1, col2 = st.columns(2)
                with col1:
                    low_thresh = st.slider("Low Threshold", 0, 255, 50, 5, key="pipeline_canny_low")
                with col2:
                    high_thresh = st.slider("High Threshold", 0, 255, 150, 5, key="pipeline_canny_high")
                
                params = {"low_threshold": low_thresh, "high_threshold": high_thresh}
            else:
                params = {}
        
        elif operation_category == "Morphology":
            operation = st.selectbox(
                "Select morphological operation:",
                ["Erosion", "Dilation", "Opening", "Closing"]
            )
            
            col1, col2 = st.columns(2)
            with col1:
                kernel_size = st.slider("Kernel Size", 3, 15, 5, 2, key="pipeline_morph_kernel")
            with col2:
                if operation in ["Erosion", "Dilation"]:
                    iterations = st.slider("Iterations", 1, 10, 1, key="pipeline_morph_iter")
                else:
                    iterations = 1
                    st.write("Single iteration for opening/closing")
            
            params = {"kernel_size": kernel_size, "iterations": iterations}
        
        elif operation_category == "Geometric":
            operation = st.selectbox(
                "Select geometric transform:",
                ["Rotation", "Scaling", "Flip Horizontal", "Flip Vertical"]
            )
            
            if operation == "Rotation":
                angle = st.slider("Rotation Angle", -180, 180, 0, 5, key="pipeline_rotation")
                params = {"angle": angle}
                
            elif operation == "Scaling":
                scale_factor = st.slider("Scale Factor", 0.1, 3.0, 1.0, 0.1, key="pipeline_scale")
                params = {"scale_factor": scale_factor}
                
            else:  # Flipping
                params = {}
        
        elif operation_category == "Noise & Denoising":
            operation = st.selectbox(
                "Select operation:",
                ["Add Gaussian Noise", "Add Salt & Pepper", "Median Denoising", "Bilateral Denoising"]
            )
            
            if operation in ["Add Gaussian Noise", "Add Salt & Pepper"]:
                noise_amount = st.slider("Noise Amount", 0.01, 0.3, 0.1, 0.01, key="pipeline_noise")
                params = {"noise_amount": noise_amount}
                
            elif operation == "Median Denoising":
                kernel_size = st.slider("Kernel Size", 3, 15, 5, 2, key="pipeline_denoise_kernel")
                params = {"kernel_size": kernel_size}
                
            elif operation == "Bilateral Denoising":
                col1, col2, col3 = st.columns(3)
                with col1:
                    d = st.slider("Diameter", 5, 25, 9, 2, key="pipeline_denoise_d")
                with col2:
                    sigma_color = st.slider("Sigma Color", 10, 150, 75, 5, key="pipeline_denoise_sc")
                with col3:
                    sigma_space = st.slider("Sigma Space", 10, 150, 75, 5, key="pipeline_denoise_ss")
                
                params = {"d": d, "sigma_color": sigma_color, "sigma_space": sigma_space}
        
        # Add step to pipeline
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button("➕ Add to Pipeline"):
                step = {
                    "category": operation_category,
                    "operation": operation,
                    "params": params
                }
                
                # Apply the operation
                try:
                    st.session_state.pipeline_image = apply_pipeline_step(
                        st.session_state.pipeline_image, step
                    )
                    st.session_state.pipeline_steps.append(step)
                    st.success(f"✅ Added: {operation}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error applying {operation}: {str(e)}")
        
        # Display current pipeline
        if st.session_state.pipeline_steps:
            st.subheader("📋 Current Pipeline")
            
            # Show pipeline steps
            for i, step in enumerate(st.session_state.pipeline_steps, 1):
                with st.expander(f"Step {i}: {step['operation']}", expanded=False):
                    st.write(f"**Category:** {step['category']}")
                    st.write(f"**Operation:** {step['operation']}")
                    if step['params']:
                        st.write("**Parameters:**")
                        for param, value in step['params'].items():
                            st.write(f"  - {param}: {value}")
            
            # Display results
            st.subheader("🖼️ Pipeline Results")
            display_images(image, st.session_state.pipeline_image, ["Original", "Processed"])
            
            # Image statistics comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Original Image:**")
                st.write(f"Shape: {image.shape}")
                st.write(f"Data type: {image.dtype}")
                st.write(f"Mean intensity: {np.mean(image):.1f}")
                st.write(f"Std deviation: {np.std(image):.1f}")
            
            with col2:
                st.write("**Processed Image:**")
                st.write(f"Shape: {st.session_state.pipeline_image.shape}")
                st.write(f"Data type: {st.session_state.pipeline_image.dtype}")
                st.write(f"Mean intensity: {np.mean(st.session_state.pipeline_image):.1f}")
                st.write(f"Std deviation: {np.std(st.session_state.pipeline_image):.1f}")
            
            # Save options
            st.subheader("💾 Save Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Download processed image
                download_image(st.session_state.pipeline_image, "pipeline_result.png")
            
            with col2:
                # Save pipeline as text
                if st.button("📄 Export Pipeline"):
                    pipeline_text = generate_pipeline_code(st.session_state.pipeline_steps)
                    st.download_button(
                        label="Download Pipeline Code",
                        data=pipeline_text,
                        file_name="image_pipeline.py",
                        mime="text/plain"
                    )
            
            with col3:
                # Show pipeline summary
                if st.button("📊 Pipeline Summary"):
                    show_pipeline_summary(st.session_state.pipeline_steps, image, st.session_state.pipeline_image)
        
        else:
            st.info("👆 Add operations to your pipeline to see results")
        
        # Preset pipelines
        st.subheader("🎯 Preset Pipelines")
        
        preset_col1, preset_col2, preset_col3 = st.columns(3)
        
        with preset_col1:
            if st.button("📷 Photo Enhancement"):
                apply_preset_pipeline("photo_enhancement", image)
        
        with preset_col2:
            if st.button("🔍 Edge Analysis"):
                apply_preset_pipeline("edge_analysis", image)
        
        with preset_col3:
            if st.button("🧹 Noise Cleanup"):
                apply_preset_pipeline("noise_cleanup", image)
        
        # Tutorial section
        if st.checkbox("📚 Pipeline Design Tutorial"):
            show_pipeline_tutorial()
    
    else:
        st.info("👆 Please upload an image or click 'Use Default Image' to begin building your pipeline.")
    
    # Key takeaways
    st.subheader("🎯 Key Takeaways")
    st.write("""
    - Image processing pipelines combine multiple operations sequentially
    - Operation order significantly affects the final result
    - Parameter tuning is crucial for optimal pipeline performance
    - Different applications require different pipeline designs
    - Documentation and reproducibility are essential for complex pipelines
    """)

def apply_pipeline_step(image, step):
    """Apply a single pipeline step to an image."""
    operation = step['operation']
    params = step['params']
    
    try:
        if operation == "Brightness/Contrast":
            return adjust_brightness_contrast(image, params['brightness'], params['contrast'])
        
        elif operation == "Negative":
            return apply_negative(image)
        
        elif operation == "Threshold":
            return apply_threshold(image, params['threshold_value'], params['threshold_type'])
        
        elif operation == "Color Space":
            if params['conversion'] == "To Grayscale":
                if len(image.shape) == 3:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                return image
            elif params['conversion'] == "To HSV":
                if len(image.shape) == 3:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                return image
        
        elif operation == "Gaussian Blur":
            return apply_gaussian_blur(image, params['kernel_size'])
        
        elif operation == "Median Blur":
            return apply_median_blur(image, params['kernel_size'])
        
        elif operation == "Bilateral Filter":
            return apply_bilateral_filter(image, params['d'], params['sigma_color'], params['sigma_space'])
        
        elif operation == "Sobel":
            return sobel_edge_detection(image)
        
        elif operation == "Laplacian":
            return laplacian_edge_detection(image)
        
        elif operation == "Canny":
            return canny_edge_detection(image, params['low_threshold'], params['high_threshold'])
        
        elif operation in ["Erosion", "Dilation", "Opening", "Closing"]:
            return morphological_operation(image, operation.lower(), params['kernel_size'], params.get('iterations', 1))
        
        elif operation == "Rotation":
            return rotate_image(image, params['angle'])
        
        elif operation == "Scaling":
            return scale_image(image, params['scale_factor'])
        
        elif operation == "Flip Horizontal":
            return flip_image(image, 1)
        
        elif operation == "Flip Vertical":
            return flip_image(image, 0)
        
        elif operation == "Add Gaussian Noise":
            return add_noise(image, "gaussian", params['noise_amount'])
        
        elif operation == "Add Salt & Pepper":
            return add_noise(image, "salt_pepper", params['noise_amount'])
        
        elif operation == "Median Denoising":
            return apply_median_blur(image, params['kernel_size'])
        
        elif operation == "Bilateral Denoising":
            return apply_bilateral_filter(image, params['d'], params['sigma_color'], params['sigma_space'])
        
        else:
            st.error(f"Unknown operation: {operation}")
            return image
    
    except Exception as e:
        st.error(f"Error in {operation}: {str(e)}")
        return image

def apply_preset_pipeline(preset_name, image):
    """Apply a preset pipeline to the image."""
    presets = {
        "photo_enhancement": [
            {"category": "Filtering", "operation": "Bilateral Filter", "params": {"d": 9, "sigma_color": 75, "sigma_space": 75}},
            {"category": "Basic Operations", "operation": "Brightness/Contrast", "params": {"brightness": 10, "contrast": 1.2}},
        ],
        "edge_analysis": [
            {"category": "Basic Operations", "operation": "Color Space", "params": {"conversion": "To Grayscale"}},
            {"category": "Filtering", "operation": "Gaussian Blur", "params": {"kernel_size": 5}},
            {"category": "Edge Detection", "operation": "Canny", "params": {"low_threshold": 50, "high_threshold": 150}},
        ],
        "noise_cleanup": [
            {"category": "Noise & Denoising", "operation": "Add Gaussian Noise", "params": {"noise_amount": 0.1}},
            {"category": "Noise & Denoising", "operation": "Median Denoising", "params": {"kernel_size": 5}},
            {"category": "Noise & Denoising", "operation": "Bilateral Denoising", "params": {"d": 9, "sigma_color": 75, "sigma_space": 75}},
        ]
    }
    
    if preset_name in presets:
        st.session_state.pipeline_steps = presets[preset_name].copy()
        st.session_state.pipeline_image = image.copy()
        
        # Apply all steps
        for step in st.session_state.pipeline_steps:
            st.session_state.pipeline_image = apply_pipeline_step(st.session_state.pipeline_image, step)
        
        st.success(f"✅ Applied preset: {preset_name.replace('_', ' ').title()}")
        st.rerun()

def generate_pipeline_code(pipeline_steps):
    """Generate Python code for the current pipeline."""
    code = """# Generated Image Processing Pipeline
import cv2
import numpy as np

def apply_pipeline(image):
    \"\"\"Apply the complete image processing pipeline.\"\"\"
    result = image.copy()
    
"""
    
    for i, step in enumerate(pipeline_steps, 1):
        operation = step['operation']
        params = step['params']
        
        code += f"    # Step {i}: {operation}\n"
        
        if operation == "Brightness/Contrast":
            code += f"    result = cv2.convertScaleAbs(result, alpha={params['contrast']}, beta={params['brightness']})\n"
        
        elif operation == "Negative":
            code += f"    result = 255 - result\n"
        
        elif operation == "Gaussian Blur":
            k = params['kernel_size']
            code += f"    result = cv2.GaussianBlur(result, ({k}, {k}), 0)\n"
        
        elif operation == "Canny":
            code += f"    if len(result.shape) == 3:\n"
            code += f"        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)\n"
            code += f"    result = cv2.Canny(result, {params['low_threshold']}, {params['high_threshold']})\n"
        
        # Add more operations as needed
        code += "\n"
    
    code += """    return result

# Example usage:
# processed_image = apply_pipeline(your_image)
"""
    
    return code

def show_pipeline_summary(pipeline_steps, original_image, processed_image):
    """Show a summary of the pipeline performance."""
    st.subheader("📊 Pipeline Performance Summary")
    
    # Basic statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Pipeline Statistics:**")
        st.write(f"Total steps: {len(pipeline_steps)}")
        
        # Count operations by category
        categories = {}
        for step in pipeline_steps:
            cat = step['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        for cat, count in categories.items():
            st.write(f"{cat}: {count} operations")
    
    with col2:
        st.write("**Image Changes:**")
        
        orig_shape = original_image.shape
        proc_shape = processed_image.shape
        
        st.write(f"Original shape: {orig_shape}")
        st.write(f"Final shape: {proc_shape}")
        
        if orig_shape == proc_shape:
            st.success("✅ Shape preserved")
        else:
            st.warning("⚠️ Shape changed")
        
        # Calculate simple quality metrics
        if orig_shape == proc_shape and len(orig_shape) == len(proc_shape):
            mse = np.mean((original_image.astype(float) - processed_image.astype(float)) ** 2)
            st.write(f"MSE: {mse:.2f}")

def show_pipeline_tutorial():
    """Show tutorial for pipeline design."""
    st.subheader("🎓 Pipeline Design Best Practices")
    
    with st.expander("1. Operation Order Matters"):
        st.write("""
        **Why order is important:**
        - Noise reduction before edge detection improves results
        - Color space conversion should happen early if needed
        - Morphological operations work best on binary images
        
        **Example:** For edge detection pipeline:
        1. Convert to grayscale (if needed)
        2. Apply Gaussian blur (noise reduction)
        3. Apply edge detection (Canny/Sobel)
        """)
    
    with st.expander("2. Parameter Tuning"):
        st.write("""
        **Guidelines for parameter selection:**
        - Start with default values and adjust gradually
        - Consider your image characteristics (noise level, contrast, etc.)
        - Test with multiple images to ensure robustness
        
        **Common parameters:**
        - Blur kernel size: 3-15 pixels (odd numbers only)
        - Canny thresholds: 50-150 for low, 100-200 for high
        - Morphology kernel: 3-7 pixels for most applications
        """)
    
    with st.expander("3. Common Pipeline Patterns"):
        st.write("""
        **Preprocessing Pipeline:**
        1. Noise reduction (Gaussian/Bilateral blur)
        2. Contrast enhancement
        3. Color space conversion (if needed)
        
        **Feature Extraction Pipeline:**
        1. Preprocessing
        2. Edge detection or texture analysis
        3. Morphological operations for cleanup
        
        **Restoration Pipeline:**
        1. Noise addition (for testing)
        2. Denoising filters
        3. Sharpening (if needed)
        """)
    
    with st.expander("4. Quality Assessment"):
        st.write("""
        **How to evaluate your pipeline:**
        - Visual inspection of results
        - Compare before/after statistics
        - Test with different input images
        - Measure processing time for real-time applications
        
        **Key metrics to consider:**
        - Signal-to-noise ratio improvement
        - Edge preservation
        - Processing speed
        - Parameter sensitivity
        """)
