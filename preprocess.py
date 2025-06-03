import cv2
import numpy as np

def detect_skin_optimized(image):
    """Phát hiện vùng da sử dụng NumPy vectorization trên không gian màu YCrCb."""
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Cr, Cb = ycrcb[:, :, 1], ycrcb[:, :, 2]

    # Áp dụng ngưỡng màu trực tiếp với NumPy
    mask = (Cr >= 133) & (Cr <= 173) & (Cb >= 77) & (Cb <= 127)
    skin_mask = np.uint8(mask) * 255

    # Giảm nhiễu bằng phép giãn nở đơn giản (dilation thay vì opening)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)

    # Tính tỷ lệ vùng da
    skin_ratio = np.sum(skin_mask > 0) / (skin_mask.shape[0] * skin_mask.shape[1])
    return skin_mask, skin_ratio

def check_image_quality_fast(image):
    """Kiểm tra chất lượng ảnh bằng gradient Sobel thay vì phương sai Laplacian."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    sharpness = np.mean(np.abs(sobel_x)) + np.mean(np.abs(sobel_y))
    return sharpness > 20  # Ngưỡng động cho chất lượng ảnh

def zoom_to_roi(image, skin_mask, zoom_factor=1.2):
    """Phóng to vùng da được phát hiện để tập trung vào khu vực quan trọng."""
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

def preprocess_image(image_path, target_size=224, resize_factor=0.5):
    """Tiền xử lý ảnh: giảm kích thước sớm, phát hiện da, kiểm tra chất lượng và chuẩn hóa."""
    image = cv2.imread(image_path, cv2.IMREAD_REDUCED_COLOR_2)  # Đọc ảnh giảm kích thước ngay từ đầu
    if image is None:
        raise ValueError(f"Không thể đọc ảnh từ {image_path}.")

    # Giảm kích thước ảnh nhanh hơn
    new_dim = (int(image.shape[1] * resize_factor), int(image.shape[0] * resize_factor))
    small_image = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)

    # Phát hiện vùng da sử dụng phương pháp tối ưu hóa
    skin_mask, skin_ratio = detect_skin_optimized(small_image)

    # Kiểm tra chất lượng ảnh
    is_clear = check_image_quality_fast(small_image)

    # Nếu ảnh không rõ hoặc vùng da nhỏ, phóng to vùng da quan trọng
    if not is_clear or skin_ratio < 0.3:
        image = zoom_to_roi(image, skin_mask)

    # Resize ảnh về kích thước mong muốn và chuẩn hóa
    resized_image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
    normalized_image = resized_image.astype(np.float32) / 255.0

    return normalized_image, skin_mask

