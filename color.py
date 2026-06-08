from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from ultralytics import YOLO

import cv2
import numpy as np
import joblib

from skimage.color import rgb2lab, rgb2hsv

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# LOAD MODELS
# ==========================
modelxg = "xgb_regression.pkl"
modelyolo = "best.pt"
yolo_model = YOLO(modelyolo)
xgb_model = joblib.load(modelxg)

# ==========================
# FEATURE EXTRACTION
# ==========================

def extract_features(img):

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    lab = rgb2lab(img_rgb)
    hsv = rgb2hsv(img_rgb)

    L = np.mean(lab[:,:,0])
    A = np.mean(lab[:,:,1])
    B = np.mean(lab[:,:,2])

    H = np.mean(hsv[:,:,0])
    S = np.mean(hsv[:,:,1])
    V = np.mean(hsv[:,:,2])

    return [L, A, B, H, S, V]

# ==========================
# API
# ==========================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()    

    npimg = np.frombuffer(contents, np.uint8)

    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # YOLO
    results = yolo_model(image)

    if len(results[0].boxes) == 0:
        return {
            "status": "not_detected"
        }

    box = results[0].boxes[0]

    cls = int(box.cls[0])

    class_name = results[0].names[cls]

    x1,y1,x2,y2 = map(
        int,
        box.xyxy[0]
    )

    roi = image[y1:y2, x1:x2]

    if roi.size == 0:
        return {
            "status": "not_detected"
    }

    if class_name.lower() == "negative":

        return {
            "status":"negative"
        }

    features = extract_features(roi)

    prediction = xgb_model.predict(
        [features]
    )[0]

    return {
        "status":"positive",
        "dna_ng": float(prediction)
    }

# ==========================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
