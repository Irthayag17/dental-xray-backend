from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import io, base64, tempfile, os
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np
import pathlib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
model = YOLO(MODEL_PATH)

CLASS_NAMES = ["Caries", "Crown", "Missing_Teeth", "Periapical_Lesion", "Root_Canal_Treatment", "Bone_Loss"]

@app.get("/api/health")
def health():
    return {"status": "ok", "model": "YOLOv8s Dental Detector"}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.save(tmp.name)
        tmp_path = tmp.name

    results = model.predict(source=tmp_path, conf=0.4, imgsz=640, verbose=False)
    os.unlink(tmp_path)

    result = results[0]
    annotated = result.plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(annotated_rgb)

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()  # ← no prefix here

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        detections.append({
            "class": CLASS_NAMES[cls_id],
            "confidence": round(conf, 3),
        })

    return JSONResponse({
        "image": img_b64,          # ← removed data:image/jpeg;base64, prefix
        "detections": detections,
        "count": len(detections)
    })

# Serve React frontend (must be LAST, after all routes)
STATIC = pathlib.Path(__file__).parent.parent / "static"
if STATIC.exists():
    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")