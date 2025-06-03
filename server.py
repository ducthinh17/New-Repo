from fastapi import FastAPI, UploadFile, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from preprocess import preprocess_image
from process import detect_and_classify
from pyngrok import ngrok
from PIL import Image, ImageDraw, ImageFont


import os
import uvicorn
import time
import base64
import io
import cv2

# Khởi tạo ứng dụng FastAPI
app = FastAPI()
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/upload_image')
async def process_image(image: UploadFile = File(...)):
    """
    API để tải lên, tiền xử lý và phát hiện mụn trên ảnh da.
    """
    try:
        start_time = time.time()
        filename = image.filename

        # Đường dẫn lưu ảnh tải lên
        filepath = os.path.join('./uploads', filename)
        with open(filepath, 'wb') as buffer:
            buffer.write(await image.read())

        # Tiền xử lý ảnh (resize, chuẩn hóa, phát hiện da)
        normalized_image, skin_mask = preprocess_image(filepath, target_size=800)

        processed_image_path = os.path.join('./uploads', f"processed_{filename}")

        # Chuyển đổi ảnh chuẩn hóa từ float32 về uint8 và lưu lại
        cv2.imwrite(processed_image_path, (normalized_image * 255).astype("uint8"))

        # Phát hiện và phân loại mụn
        bounding_boxes, annotated_image, total_acnes = detect_and_classify(processed_image_path)

        # Chuyển ảnh chú thích thành base64 để trả về
        output_buffer = io.BytesIO()
        annotated_image.save(output_buffer, format="PNG")
        output_image_base64 = base64.b64encode(output_buffer.getvalue()).decode("utf-8")

        # Xóa ảnh tạm để tiết kiệm bộ nhớ
        os.remove(filepath)
        os.remove(processed_image_path)

        # Chuẩn bị dữ liệu phản hồi
        response_data = {
            'bounding_boxes': bounding_boxes,
            'total_acnes': total_acnes,
            'output_image': output_image_base64
        }
        response_data['time_taken'] = round(time.time() - start_time, 2)

        return JSONResponse(content=jsonable_encoder(response_data), status_code=200)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return JSONResponse(content=jsonable_encoder({'error': str(e), 'details': error_details}), status_code=500)

if __name__ == '__main__':
    port = 8001
    os.makedirs('./uploads', exist_ok=True)
    public_url = ngrok.connect(port)
    print(f" * Ngrok Public URL: \"{public_url}\" -> \"http://127.0.0.1:{port}/\"")
    uvicorn.run(app, host='0.0.0.0', port=port)
