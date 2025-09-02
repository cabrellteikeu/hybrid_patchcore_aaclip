from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageDraw
import torch
import io
import numpy as np
import cv2
import os
from torchvision import transforms
from anomalib.models import Patchcore
import io, yaml
from pathlib import Path

CFG = yaml.safe_load(Path("config/hybrid_config.yaml").read_text(encoding="utf-8"))
HEATMAP_DIR = CFG["patchcore"]["heatmap_dir"]
CKPT_PATH = CFG["patchcore"]["ckpt_path"]
IMAGE_SIZE = CFG["patchcore"]["image_size"]
THRESHOLD = CFG["patchcore"]["threshold"]

app = FastAPI()

# Konfiguration


os.makedirs(HEATMAP_DIR, exist_ok=True)

# Modell laden
model = Patchcore.load_from_checkpoint(CKPT_PATH)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Transform definieren
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

def visualize_anomalies(image, anomaly_map, threshold=0.5):
    """Visualize anomalies on the image."""
    image = image.copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for y in range(anomaly_map.shape[0]):
        for x in range(anomaly_map.shape[1]):
            if anomaly_map[y, x] > threshold:
                draw.rectangle([x * (width // anomaly_map.shape[1]), y * (height // anomaly_map.shape[0]),
                                (x + 1) * (width // anomaly_map.shape[1]), (y + 1) * (height // anomaly_map.shape[0])],
                               outline="red", width=2)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Bild laden & vorbereiten
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)

        # Robust extrahieren
        if isinstance(output, dict):
            score = float(output.get("pred_scores", torch.tensor([0.0])).item())
            anomaly_map = output.get("anomaly_maps", torch.zeros(IMAGE_SIZE))
        elif hasattr(output, 'anomaly_map'):
            score = float(output.pred_score.item())
            anomaly_map = output.anomaly_map
        else:
            score = float(output.item())
            anomaly_map = torch.zeros(IMAGE_SIZE)

        # Ensure anomaly_map is a 2D array
        anomaly_map = anomaly_map.squeeze().cpu().numpy()

        pred_label = 1 if score > THRESHOLD else 0

        # Visualisierung der Anomalien
        if pred_label == 1:
            visualized_image = visualize_anomalies(image, anomaly_map, THRESHOLD)

            # Speichern des visualisierten Bildes
            filename = os.path.splitext(file.filename)[0]
            heatmap_path = os.path.join(HEATMAP_DIR, f"{filename}_overlay.jpg")
            visualized_image.save(heatmap_path)

            # Bild als Antwort zurückgeben
            response = FileResponse(heatmap_path, media_type="image/jpeg")
            response.headers["X-Prediction-Score"] = str(score)
            return response

        return JSONResponse(content={
            "prediction": pred_label,
            "score": score
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "API is running and ready"}

# Starte mit: uvicorn run_inference_api:app --reload --host 127.0.0.1 --port 8000
