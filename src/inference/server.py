"""
    Inference server for Synthetic Inspector. 

    Loads the INT8 quantized defect classifier and exposes an HTTP API. 

    The model file at MODEL_INT8_PATH is a PyTorch state_dict for a dynamically-quantized ResNet-18. 
    To load it:
        1. Build the float ResNet-18 (build_model, same as train.py)
        2. Apply torch.quantization.quantize_dynamic to it.
        3. Load the saved state_dict into the quantized model

    The API:
        POST /predict multipart/form-data, field "file = image
                        returns {"class": str, "confidence": float,
                        "is_defect": bool}

        GET /health returns {"status": "ok", "model_loaded": bool}
        GET /classes returns {"classes": [...]}

    Run:
        python -m src.inference.server
        uvicorn src.inference.server:app --host 0.0.0.0 --port 8080
"""

import io 
from pathlib import Path 

import torch 
import torch.nn as nn 
import torch.quantization 
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms, models 

from src.config import (
    MODEL_INT8_PATH,
    IMAGE_SIZE,
    SERVER_HOST,
    SERVER_PORT,
    CONFIDENCE_THRESHOLD,
    DEFECT_CLASSES,
    IDX_TO_CLASS,
)


def build_quantized_model(num_classes: int) -> nn.Module:
    """
        Rebuild the same quantized architecture that train.py saved. 
        ...
    """

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.eval()
    quantized = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )   
    return quantized


def load_model() -> nn.Module:
    if not MODEL_INT8_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_INT8_PATH}. Run train.py first."
        )

    model = build_quantized_model(len(DEFECT_CLASSES))
    state_dict = torch.load(MODEL_INT8_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# Preprocessing 
_preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
        Decode bytes to PIL, convert to RGB, apply transform, add batch dim.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    tensor = _preprocess(image).unsqueeze(0)
    return tensor


app = FastAPI(title="Synthetic Inspector", version="0.1.0")
_model = None

def get_model() -> nn.Module:
    global _model
    if _model is None:
        _model = load_model()
    return _model



@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}

@app.get("/classes")
def classes() -> dict:
    return {"classes": DEFECT_CLASSES}

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    tensor = preprocess_image(image_bytes)
    model = get_model()

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, class_idx = probs.max(dim=1)

    class_idx_int = int(class_idx.item())
    confidence_float = float(confidence.item())
    class_name = IDX_TO_CLASS[class_idx_int]

    return JSONResponse({
        "class": class_name,
        "confidence": round(confidence_float, 4),
        "is_defect": class_name != "good" and confidence_float >= CONFIDENCE_THRESHOLD,
    })  


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.inference.server:app", host=SERVER_HOST, port=SERVER_PORT, reload=False)
    