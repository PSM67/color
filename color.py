from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import joblib
from skimage.color import rgb2lab, rgb2hsv

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://psm67.github.io"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# CONFIGURATION & LOAD MODELS
# ==========================
# ⚠️ Update this list to match your exact custom YOLO training classes in order!
CLASSES = ["negative", "positive"] 

modelxg = "xgb_regression.pkl"
modelyolo = "best.onnx"  # Pointing to your converted .onnx model

# Load YOLO via OpenCV's DNN engine (Uses minimal RAM compared to PyTorch)
yolo_net = cv2.dnn.readNetFromONNX(modelyolo)
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

    h_orig, w_orig = image.shape[:2]

    # Preprocess image into a 640x640 input blob for YOLO
    blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(640, 640), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    
    # Run Inference
    outputs = yolo_net.forward()
    
    # Post-process YOLOv8 output matrix structure
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]
    
    boxes = []
    scores = []
    class_ids = []
    
    x_scale = w_orig / 640
    y_scale = h_orig / 640

    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        cls = np.argmax(classes_scores)
        max_score = classes_scores[cls]
        
        # Filter detections by confidence threshold
        if max_score >= 0.25:
            cx, cy, w, h = outputs[0][i][0], outputs[0][i][1], outputs[0][i][2], outputs[0][i][3]
            
            # Convert center coords to top-left corner box coordinates scaled to original image
            x1 = int((cx - 0.5 * w) * x_scale)
            y1 = int((cy - 0.5 * h) * y_scale)
            w = int(w * x_scale)
            h = int(h * y_scale)
            
            boxes.append([x1, y1, w, h])
            scores.append(float(max_score))
            class_ids.append(int(cls))

    # Apply Non-Maximum Suppression to clear duplicate overlapping windows
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.25, nms_threshold=0.45)

    if len(indices) == 0:
        return {
            "status": "not_detected"
        }

    # Extract data for the highest confidence detection
    best_idx = indices[0]
    if isinstance(best_idx, (list, np.ndarray, tuple)):
        best_idx = best_idx[0]

    x1, y1, w, h = boxes[best_idx]
    cls = class_ids[best_idx]
    class_name = CLASSES[cls]

    # Clip boundary dimensions inside the original image dimensions
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_orig, x1 + w), min(h_orig, y1 + h)

    roi = image[y1:y2, x1:x2]

    if roi.size == 0:
        return {
            "status": "not_detected"
        }

    if class_name.lower() == "negative":
        return {
            "status": "negative"
        }

    features = extract_features(roi)

    prediction = xgb_model.predict([features])[0]

    return {
        "status": "positive",
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
