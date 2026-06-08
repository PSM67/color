import os
import cv2
import json
import pickle
import joblib
import numpy as np
import xgboost as xgb

from fastapi import FastAPI, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = r"D:\work\Aj Ying\backend\dna_concentration.pkl"
model = joblib.load(MODEL_PATH)

def compute_channel_stats(roi_img, prefix):
    B, G, R = cv2.split(roi_img)

    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    lab = cv2.cvtColor(roi_img, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)

    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

    G_idx = np.where(G == 0, 1, G)
    B_idx = np.where(B == 0, 1, B)

    features = {
        f'{prefix}_R_mean': np.mean(R),
        f'{prefix}_G_mean': np.mean(G),
        f'{prefix}_B_mean': np.mean(B),
        f'{prefix}_R_std': np.std(R),
        f'{prefix}_G_std': np.std(G),
        f'{prefix}_B_std': np.std(B),
        f'{prefix}_H_mean': np.mean(H),
        f'{prefix}_S_mean': np.mean(S),
        f'{prefix}_V_mean': np.mean(V),
        f'{prefix}_H_std': np.std(H),
        f'{prefix}_S_std': np.std(S),
        f'{prefix}_V_std': np.std(V),
        f'{prefix}_L_mean': np.mean(L),
        f'{prefix}_a_mean': np.mean(a),
        f'{prefix}_b_mean': np.mean(b),
        f'{prefix}_L_std': np.std(L),
        f'{prefix}_a_std': np.std(a),
        f'{prefix}_b_std': np.std(b),
        f'{prefix}_gray_mean': np.mean(gray),
        f'{prefix}_gray_std': np.std(gray),
        f'{prefix}_R_over_G': np.mean(R / G_idx),
        f'{prefix}_B_over_G': np.mean(B / G_idx),
        f'{prefix}_R_over_B': np.mean(R / B_idx),
    }
    return features


@app.get("/", response_class=HTMLResponse)
async def home():
    if not os.path.exists("index.html"):
        return "<h1>Backend Running</h1>"
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


# ==========================
# API
# ==========================

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    rois: str = Form(...)
):
    if not image or not rois:
        return {"error": "Missing image or ROI coordinates"}

    file = await image.read()

    try:
        rois_coord = json.loads(rois)
        file_bytes = np.frombuffer(file, np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image data"}

        def crop(r):
            return img[
                int(r['y1']):int(r['y2']),
                int(r['x1']):int(r['x2'])
            ]

        roi_band = crop(rois_coord['result'])
        roi_bg = crop(rois_coord['bg'])
        roi_upper = crop(rois_coord['upper'])

        if roi_band.size == 0: return {"error": "Invalid Result ROI"}
        if roi_bg.size == 0: return {"error": "Invalid Background ROI"}
        if roi_upper.size == 0: return {"error": "Invalid Upper ROI"}

        f_band = compute_channel_stats(roi_band, 'band')
        f_bg = compute_channel_stats(roi_bg, 'bg')
        f_upper = compute_channel_stats(roi_upper, 'upper')

        all_features = {}
        all_features.update(f_band)
        all_features.update(f_bg)
        all_features.update(f_upper)

        metrics = ['R_mean', 'G_mean', 'B_mean', 'H_mean', 'S_mean', 'V_mean', 'L_mean', 'a_mean', 'b_mean', 'gray_mean']

        for m in metrics:
            all_features[f'delta_{m}'] = f_band[f'band_{m}'] - f_bg[f'bg_{m}']
            all_features[f'band_minus_upper_{m}'] = f_band[f'band_{m}'] - f_upper[f'upper_{m}']

        feature_order = [
            'band_R_mean', 'band_G_mean', 'band_B_mean', 'band_R_std', 'band_G_std', 'band_B_std',
            'band_H_mean', 'band_S_mean', 'band_V_mean', 'band_H_std', 'band_S_std', 'band_V_std',
            'band_L_mean', 'band_a_mean', 'band_b_mean', 'band_L_std', 'band_a_std', 'band_b_std',
            'band_gray_mean', 'band_gray_std', 'band_R_over_G', 'band_B_over_G', 'band_R_over_B',
            'bg_R_mean', 'bg_G_mean', 'bg_B_mean', 'bg_R_std', 'bg_G_std', 'bg_B_std',
            'bg_H_mean', 'bg_S_mean', 'bg_V_mean', 'bg_H_std', 'bg_S_std', 'bg_V_std',
            'bg_L_mean', 'bg_a_mean', 'bg_b_mean', 'bg_L_std', 'bg_a_std', 'bg_b_std',
            'bg_gray_mean', 'bg_gray_std', 'bg_R_over_G', 'bg_B_over_G', 'bg_B_over_R' if 'bg_B_over_R' in all_features else 'bg_R_over_B', # Safe tracking fallback
            'upper_R_mean', 'upper_G_mean', 'upper_B_mean', 'upper_R_std', 'upper_G_std', 'upper_B_std',
            'upper_H_mean', 'upper_S_mean', 'upper_V_mean', 'upper_H_std', 'upper_S_std', 'upper_V_std',
            'upper_L_mean', 'upper_a_mean', 'upper_b_mean', 'upper_L_std', 'upper_a_std', 'upper_b_std',
            'upper_gray_mean', 'upper_gray_std', 'upper_R_over_G', 'upper_B_over_G', 'upper_R_over_B',
            'delta_R_mean', 'band_minus_upper_R_mean', 'delta_G_mean', 'band_minus_upper_G_mean',
            'delta_B_mean', 'band_minus_upper_B_mean', 'delta_H_mean', 'band_minus_upper_H_mean',
            'delta_S_mean', 'band_minus_upper_S_mean', 'delta_V_mean', 'band_minus_upper_V_mean',
            'delta_L_mean', 'band_minus_upper_L_mean', 'delta_a_mean', 'band_minus_upper_a_mean',
            'delta_b_mean', 'band_minus_upper_b_mean', 'delta_gray_mean', 'band_minus_upper_gray_mean'
        ]

        X_input = np.array(
            [[all_features[name] for name in feature_order]],
            dtype=np.float32
        )

        if model is None:
            return {"error": "Model structure error"}

        # 🛠️ FIXED: Check if the loaded object is actually a dictionary wrapper
        actual_model = model
        if isinstance(model, dict):
            # Check for common dictionary keys used when exporting models
            for key in ['model', 'regressor', 'model_state', 'xgb_model', 'estimator']:
                if key in model:
                    actual_model = model[key]
                    break
            else:
                # If no standard key is matched, return the dictionary keys directly to your frontend screen
                return {
                    "error": f"The .pkl file is a dictionary wrapper. Available keys are: {list(model.keys())}. Extract the true model artifact using the correct key."
                }

        # Run prediction on the extracted model object
        prediction = actual_model.predict(X_input)[0]

        return {
            "status": "success",
            "value": float(prediction)
        }

    except Exception as e:
        return {"error": f"Internal processing error: {str(e)}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
