import cv2
import numpy as np

def adjust_brightness_contrast(image, brightness=0, contrast=1.0):
    """Adjust brightness and contrast of an image."""
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

def apply_negative(image):
    """Apply negative transformation to an image."""
    return 255 - image

def apply_threshold(image, threshold_value, threshold_type='binary'):
    """Apply thresholding to an image."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    threshold_types = {
        'binary': cv2.THRESH_BINARY,
        'binary_inv': cv2.THRESH_BINARY_INV,
        'truncate': cv2.THRESH_TRUNC,
        'to_zero': cv2.THRESH_TOZERO,
        'to_zero_inv': cv2.THRESH_TOZERO_INV
    }
    
    _, thresh = cv2.threshold(gray, threshold_value, 255, threshold_types[threshold_type])
    return thresh

def apply_gaussian_blur(image, kernel_size):
    """Apply Gaussian blur to an image."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_median_blur(image, kernel_size):
    """Apply median blur to an image."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(image, kernel_size)

def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """Apply bilateral filter to an image."""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def apply_custom_kernel(image, kernel):
    """Apply custom kernel to an image."""
    return cv2.filter2D(image, -1, kernel)

def add_noise(image, noise_type='gaussian', amount=0.1):
    """Add noise to an image."""
    if noise_type == 'gaussian':
        noise = np.random.normal(0, amount * 255, image.shape)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    elif noise_type == 'salt_pepper':
        noisy = image.copy()
        # Salt noise
        salt = np.random.random(image.shape[:2])
        noisy[salt < amount/2] = 255
        # Pepper noise
        pepper = np.random.random(image.shape[:2])
        noisy[pepper < amount/2] = 0
        return noisy
    
    return image

def sobel_edge_detection(image):
    """Apply Sobel edge detection."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    
    return np.uint8(np.clip(sobel_combined, 0, 255))

def laplacian_edge_detection(image):
    """Apply Laplacian edge detection."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return np.uint8(np.clip(np.absolute(laplacian), 0, 255))

def canny_edge_detection(image, low_threshold=50, high_threshold=150):
    """Apply Canny edge detection."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    return cv2.Canny(gray, low_threshold, high_threshold)

def morphological_operation(image, operation, kernel_size=5, iterations=1):
    """Apply morphological operations."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Ensure binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    operations = {
        'erosion': cv2.MORPH_ERODE,
        'dilation': cv2.MORPH_DILATE,
        'opening': cv2.MORPH_OPEN,
        'closing': cv2.MORPH_CLOSE
    }
    
    if operation in ['erosion', 'dilation']:
        return cv2.morphologyEx(binary, operations[operation], kernel, iterations=iterations)
    else:
        return cv2.morphologyEx(binary, operations[operation], kernel)

def rotate_image(image, angle):
    """Rotate image by given angle."""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
    
    return rotated

def scale_image(image, scale_factor):
    """Scale image by given factor."""
    height, width = image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

def flip_image(image, flip_code):
    """Flip image. flip_code: 0=vertical, 1=horizontal, -1=both"""
    return cv2.flip(image, flip_code)
