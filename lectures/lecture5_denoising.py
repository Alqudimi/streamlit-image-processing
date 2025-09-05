import streamlit as st
import cv2
import numpy as np
from utils.image_utils import load_image, display_images
from utils.processing import add_noise, apply_median_blur, apply_bilateral_filter

def show():
    st.header("المحاضرة الخامسة: الضوضاء وتقنيات إزالة الضوضاء")
    
    # Theory section
    st.subheader("النظرية")
    st.write("""
    ضوضاء الصور تأتي من قيود المستشعر وأخطاء النقل والعوامل البيئية.
    الضوضاء الغاوسية تظهر كتغيرات عشوائية في الكثافة عبر جميع البكسلات بانتظام.
    ضوضاء الملح والفلفل تنشئ بكسلات بيضاء وسوداء معزولة موزعة عشوائياً.
    مرشحات إزالة الضوضاء تزيل الضوضاء مع الحفاظ على ميزات الصورة المهمة والحواف.
    """)
    
    st.subheader("العرض التفاعلي")
    
    # Load image
    image = load_image()
    
    if image is not None:
        # Demo type selection
        demo_type = st.selectbox(
            "اختر العرض التوضيحي:",
            ["إضافة ضوضاء وإزالتها", "مقارنة الضوضاء", "فعالية المرشحات"]
        )
        
        if demo_type == "Add Noise & Denoise":
            st.subheader("🎭 Add Noise and Apply Denoising")
            
            # Noise parameters
            col1, col2 = st.columns(2)
            
            with col1:
                noise_type = st.selectbox("Noise Type", ["gaussian", "salt_pepper"])
                noise_amount = st.slider("Noise Amount", 0.0, 0.3, 0.1, 0.01)
            
            with col2:
                filter_type = st.selectbox("Denoising Filter", ["median", "bilateral", "gaussian"])
                if filter_type in ["median", "gaussian"]:
                    filter_size = st.slider("Filter Size", 3, 15, 5, 2)
                else:  # bilateral
                    d = st.slider("Diameter", 5, 25, 9, 2)
                    sigma_color = st.slider("Sigma Color", 10, 150, 75, 5)
                    sigma_space = st.slider("Sigma Space", 10, 150, 75, 5)
            
            # Add noise
            noisy_image = add_noise(image, noise_type, noise_amount)
            
            # Apply denoising
            if filter_type == "median":
                denoised = apply_median_blur(noisy_image, filter_size)
            elif filter_type == "bilateral":
                denoised = apply_bilateral_filter(noisy_image, d, sigma_color, sigma_space)
            else:  # gaussian
                if filter_size % 2 == 0:
                    filter_size += 1
                denoised = cv2.GaussianBlur(noisy_image, (filter_size, filter_size), 0)
            
            # Display three images
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Original")
                if len(image.shape) == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    st.image(image_rgb, use_column_width=True)
                else:
                    st.image(image, use_column_width=True)
            
            with col2:
                st.subheader(f"Noisy ({noise_type})")
                if len(noisy_image.shape) == 3:
                    noisy_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)
                    st.image(noisy_rgb, use_column_width=True)
                else:
                    st.image(noisy_image, use_column_width=True)
            
            with col3:
                st.subheader(f"Denoised ({filter_type})")
                if len(denoised.shape) == 3:
                    denoised_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
                    st.image(denoised_rgb, use_column_width=True)
                else:
                    st.image(denoised, use_column_width=True)
            
            # Quality metrics
            st.subheader("📊 Quality Assessment")
            
            # Calculate PSNR (simplified)
            def calculate_psnr(original, processed):
                mse = np.mean((original.astype(float) - processed.astype(float)) ** 2)
                if mse == 0:
                    return 100
                max_pixel = 255.0
                psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
                return psnr
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Original Std Dev", f"{np.std(image):.1f}")
            
            with col2:
                noisy_psnr = calculate_psnr(image, noisy_image)
                st.metric("Noisy PSNR", f"{noisy_psnr:.1f} dB")
            
            with col3:
                denoised_psnr = calculate_psnr(image, denoised)
                st.metric("Denoised PSNR", f"{denoised_psnr:.1f} dB")
            
            if denoised_psnr > noisy_psnr:
                st.success(f"✅ Improvement: +{denoised_psnr - noisy_psnr:.1f} dB")
            else:
                st.warning(f"⚠️ Quality loss: {denoised_psnr - noisy_psnr:.1f} dB")
        
        elif demo_type == "Noise Comparison":
            st.subheader("🔍 Different Types of Noise")
            
            noise_level = st.slider("Noise Level", 0.05, 0.3, 0.15, 0.05)
            
            # Add different types of noise
            gaussian_noise = add_noise(image, "gaussian", noise_level)
            salt_pepper_noise = add_noise(image, "salt_pepper", noise_level)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Original")
                if len(image.shape) == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    st.image(image_rgb, use_column_width=True)
                else:
                    st.image(image, use_column_width=True)
                st.write("Clean reference image")
            
            with col2:
                st.subheader("Gaussian Noise")
                if len(gaussian_noise.shape) == 3:
                    gaussian_rgb = cv2.cvtColor(gaussian_noise, cv2.COLOR_BGR2RGB)
                    st.image(gaussian_rgb, use_column_width=True)
                else:
                    st.image(gaussian_noise, use_column_width=True)
                st.write("Random intensity variations")
            
            with col3:
                st.subheader("Salt & Pepper")
                if len(salt_pepper_noise.shape) == 3:
                    salt_pepper_rgb = cv2.cvtColor(salt_pepper_noise, cv2.COLOR_BGR2RGB)
                    st.image(salt_pepper_rgb, use_column_width=True)
                else:
                    st.image(salt_pepper_noise, use_column_width=True)
                st.write("Isolated black/white pixels")
            
            # Noise characteristics
            st.subheader("📈 Noise Characteristics")
            
            if len(image.shape) == 3:
                orig_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gauss_gray = cv2.cvtColor(gaussian_noise, cv2.COLOR_BGR2GRAY)
                sp_gray = cv2.cvtColor(salt_pepper_noise, cv2.COLOR_BGR2GRAY)
            else:
                orig_gray = image
                gauss_gray = gaussian_noise
                sp_gray = salt_pepper_noise
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Gaussian Noise Analysis:**")
                gauss_diff = gauss_gray.astype(float) - orig_gray.astype(float)
                st.write(f"Mean: {np.mean(gauss_diff):.2f}")
                st.write(f"Std Dev: {np.std(gauss_diff):.2f}")
                st.write(f"Distribution: Normal/Bell curve")
            
            with col2:
                st.write("**Salt & Pepper Analysis:**")
                sp_diff = sp_gray.astype(float) - orig_gray.astype(float)
                unique_vals = np.unique(sp_diff)
                st.write(f"Unique differences: {len(unique_vals)}")
                st.write(f"Extreme values: {np.min(sp_diff):.0f}, {np.max(sp_diff):.0f}")
                st.write(f"Distribution: Impulse/Sparse")
        
        elif demo_type == "Filter Effectiveness":
            st.subheader("🏆 Filter Effectiveness Comparison")
            
            # Add noise first
            noise_type = st.selectbox("Test with noise type:", ["gaussian", "salt_pepper"])
            noise_level = st.slider("Noise level", 0.05, 0.25, 0.1, 0.05)
            
            noisy = add_noise(image, noise_type, noise_level)
            
            # Apply different filters
            median_filtered = apply_median_blur(noisy, 5)
            bilateral_filtered = apply_bilateral_filter(noisy, 9, 75, 75)
            gaussian_filtered = cv2.GaussianBlur(noisy, (5, 5), 0)
            
            # Display all results
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.write("**Original**")
                if len(image.shape) == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    st.image(image_rgb, use_column_width=True)
                else:
                    st.image(image, use_column_width=True)
            
            with col2:
                st.write("**Median Filter**")
                if len(median_filtered.shape) == 3:
                    median_rgb = cv2.cvtColor(median_filtered, cv2.COLOR_BGR2RGB)
                    st.image(median_rgb, use_column_width=True)
                else:
                    st.image(median_filtered, use_column_width=True)
            
            with col3:
                st.write("**Bilateral Filter**")
                if len(bilateral_filtered.shape) == 3:
                    bilateral_rgb = cv2.cvtColor(bilateral_filtered, cv2.COLOR_BGR2RGB)
                    st.image(bilateral_rgb, use_column_width=True)
                else:
                    st.image(bilateral_filtered, use_column_width=True)
            
            with col4:
                st.write("**Gaussian Blur**")
                if len(gaussian_filtered.shape) == 3:
                    gaussian_rgb = cv2.cvtColor(gaussian_filtered, cv2.COLOR_BGR2RGB)
                    st.image(gaussian_rgb, use_column_width=True)
                else:
                    st.image(gaussian_filtered, use_column_width=True)
            
            # Effectiveness analysis
            st.subheader("📊 Filter Performance")
            
            def calculate_psnr(original, processed):
                mse = np.mean((original.astype(float) - processed.astype(float)) ** 2)
                if mse == 0:
                    return 100
                max_pixel = 255.0
                psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
                return psnr
            
            median_psnr = calculate_psnr(image, median_filtered)
            bilateral_psnr = calculate_psnr(image, bilateral_filtered)
            gaussian_psnr = calculate_psnr(image, gaussian_filtered)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Median PSNR", f"{median_psnr:.1f} dB")
            with col2:
                st.metric("Bilateral PSNR", f"{bilateral_psnr:.1f} dB")
            with col3:
                st.metric("Gaussian PSNR", f"{gaussian_psnr:.1f} dB")
            
            # Recommendations
            if noise_type == "salt_pepper":
                st.info("🎯 **Recommendation**: Median filter works best for salt & pepper noise")
            else:
                st.info("🎯 **Recommendation**: Bilateral filter preserves edges while removing Gaussian noise")
    
    else:
        st.info("👆 Please upload an image or click 'Use Default Image' to begin the demonstration.")
    
    # Key takeaways
    st.subheader("🎯 Key Takeaways")
    st.write("""
    - Different noise types require different denoising approaches
    - Median filter excels at removing salt-and-pepper noise
    - Bilateral filter preserves edges while smoothing uniform regions
    - Gaussian blur is simple but may over-smooth important details
    - Quality assessment helps evaluate denoising effectiveness
    """)
