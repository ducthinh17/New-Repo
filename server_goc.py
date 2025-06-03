from fastapi import FastAPI, UploadFile, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Built-ins
import os
import time
import io
import base64

# 3rd Party Libraries
import uvicorn
import cv2
from pyngrok import ngrok
from PIL import Image

# Local Imports
from preprocess import preprocess_image
from process import detect_and_classify

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
    Endpoint to receive an image, preprocess it, run detection + classification,
    and return bounding boxes plus a base64-encoded annotated image.
    """
    try:
        start_time = time.time()
        filename = image.filename

        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            raise ValueError("Unsupported file format. Please upload a JPG or PNG image.")

        filepath = os.path.join("./uploads", filename)
        with open(filepath, "wb") as buffer:
            buffer.write(await image.read())

        processed_image = preprocess_image(filepath, target_size=800)
        processed_image_path = os.path.join("./uploads", f"processed_{filename}")
        cv2.imwrite(processed_image_path, (processed_image * 255).astype("uint8"))

        bounding_boxes, annotated_image, total_acnes = detect_and_classify(processed_image_path)

        output_buffer = io.BytesIO()
        annotated_image.save(output_buffer, format="JPEG")
        base64_image = base64.b64encode(output_buffer.getvalue()).decode("utf-8")

        os.remove(filepath)
        os.remove(processed_image_path)

        response_data = {
            "bounding_boxes": bounding_boxes,
            "total_acnes": total_acnes,
            "output_image": base64_image,
            "time_taken": round(time.time() - start_time, 2)
        }

        return JSONResponse(content=jsonable_encoder(response_data), status_code=200)

    except Exception as e:
        return JSONResponse(content=jsonable_encoder({"error": str(e)}), status_code=500)

if __name__ == "__main__":
    os.makedirs("./uploads", exist_ok=True)
    port = 8001
    public_url = ngrok.connect(port)
    print(f' *"{public_url}" -> "http://127.0.0.1:{port}/"')
    uvicorn.run(app, host="0.0.0.0", port=port)
