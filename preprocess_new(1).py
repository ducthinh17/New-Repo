import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

cv2.setUseOptimized(True)
cv2.setNumThreads(os.cpu_count())

def detect_skin_optimized(image):
    """Tối ưu hóa phát hiện vùng da bằng kênh YCrCb."""
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Cr, Cb = cv2.split(ycrcb)[1:3]

    # Áp dụng threshold trực tiếp (dùng NumPy vectorization để tăng tốc)
    skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))

    # Giảm nhiễu nhanh bằng fastNlMeansDenoising
    skin_mask = cv2.fastNlMeansDenoising(skin_mask, h=5)

    skin_ratio = np.count_nonzero(skin_mask) / skin_mask.size
    return skin_mask, skin_ratio

def check_image_quality_fast(image):
    """Kiểm tra độ sắc nét bằng Laplacian với tối ưu hóa."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var > 100  # Ngưỡng cố định để đơn giản hóa

def zoom_to_roi_optimized(image, skin_mask, zoom_factor=1.3):
    """Cắt vùng da và phóng to nhanh bằng cv2.boundingRect."""
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        padding = max(w, h) // 4
        x1, x2 = max(0, x - padding), min(image.shape[1], x + w + padding)
        y1, y2 = max(0, y - padding), min(image.shape[0], y + h + padding)
        cropped = image[y1:y2, x1:x2]
        zoomed = cv2.resize(cropped, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        return zoomed
    return image

def preprocess_image_optimized(image_path, target_size=224, resize_factor=0.5):
    """Tiền xử lý ảnh tối ưu: giảm kích thước, phát hiện da và kiểm tra chất lượng."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh từ {image_path}.")

    small_image = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)

    # Phát hiện vùng da tối ưu
    skin_mask, skin_ratio = detect_skin_optimized(small_image)

    # Kiểm tra chất lượng ảnh nhanh hơn
    is_clear = check_image_quality_fast(small_image)

    # Nếu ảnh không rõ hoặc vùng da nhỏ, phóng to vùng ROI
    if not is_clear or skin_ratio < 0.3:
        image = zoom_to_roi_optimized(image, skin_mask)

    # Resize và chuẩn hóa ảnh
    resized_image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
    normalized_image = resized_image.astype(np.float32) / 255.0

    return normalized_image, skin_mask