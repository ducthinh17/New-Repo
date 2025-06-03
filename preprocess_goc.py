import cv2
import numpy as np

def detect_skin(image):
    """Detect skin using YCrCb color space."""
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    skin_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
    return mask, skin_ratio

def check_image_quality(image):
    """Check image clarity using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var > 100  # Threshold for clarity

def zoom_to_roi(image, zoom_factor=1.3):
    """Zoom into the region of interest by a factor."""
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2

    new_width = int(width / zoom_factor)
    new_height = int(height / zoom_factor)
    x1 = max(0, center_x - new_width // 2)
    x2 = min(width, center_x + new_width // 2)
    y1 = max(0, center_y - new_height // 2)
    y2 = min(height, center_y + new_height // 2)

    cropped = image[y1:y2, x1:x2]
    zoomed = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
    return zoomed

def preprocess_image(image_path, target_size=224):
    """Preprocess the image and normalize it."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image from {image_path}.")

    # Check for skin ratio and clarity
    _, skin_ratio = detect_skin(image)
    is_clear = check_image_quality(image)

    # Apply zoom if needed
    if not is_clear or skin_ratio < 0.3:
        image = zoom_to_roi(image, zoom_factor=1.3)

    # Resize and pad image to target size
    resized_image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
    normalized_image = resized_image.astype(np.float32) / 255.0
    return normalized_image
