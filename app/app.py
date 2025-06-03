import os
import time
import io
import base64

from fastapi import FastAPI, UploadFile, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import uvicorn
import cv2

from preprocess import preprocess_image
from process import detect_and_classify

# Tạo thư mục ./uploads nếu chưa tồn tại
os.makedirs("./uploads", exist_ok=True)

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload_image")
async def process_image(image: UploadFile = File(...)):
    """
    API endpoint để tải lên ảnh, tiền xử lý và phát hiện/phân loại mụn.
    """
    try:
        start_time = time.time()
        filename = image.filename

        # Lưu file vừa tải lên vào ./uploads/<filename>
        upload_path = os.path.join("./uploads", filename)
        with open(upload_path, "wb") as buffer:
            buffer.write(await image.read())

        # Tiền xử lý ảnh (resize, normalize, phát hiện vùng da)
        normalized_image, skin_mask = preprocess_image(upload_path, target_size=800)

        # Lưu ảnh đã chuẩn hóa thành file tạm để detect
        processed_image_path = os.path.join("./uploads", f"processed_{filename}")
        # Chuyển normalized_image (float32 trong [0,1]) về uint8
        cv2.imwrite(processed_image_path, (normalized_image * 255).astype("uint8"))

        # Chạy hàm detect_and_classify trên ảnh đã xử lý
        bounding_boxes, annotated_image, total_acnes = detect_and_classify(processed_image_path)

        # Chuyển ảnh có chú thích (PIL Image) sang base64 để trả về JSON
        output_buffer = io.BytesIO()
        annotated_image.save(output_buffer, format="PNG")
        output_image_base64 = base64.b64encode(output_buffer.getvalue()).decode("utf-8")

        # Xóa các file tạm để tiết kiệm dung lượng
        os.remove(upload_path)
        os.remove(processed_image_path)

        # Chuẩn bị payload phản hồi
        response_data = {
            "bounding_boxes": bounding_boxes,
            "total_acnes": total_acnes,
            "output_image": output_image_base64,
            "time_taken": round(time.time() - start_time, 2),
        }
        return JSONResponse(content=jsonable_encoder(response_data), status_code=200)

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        return JSONResponse(
            content=jsonable_encoder({"error": str(e), "details": error_details}),
            status_code=500,
        )


if __name__ == "__main__":
    # Khi chạy local hoặc uvicorn trực tiếp, lấy PORT từ biến môi trường (mặc định 8000)
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
