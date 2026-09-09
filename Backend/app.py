import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from backend.predictor import predict, DEVICE, SPECIES_CLASSES, FRESHNESS_CLASSES

import shutil

# ===========================================================
# APP INITIALIZATION
# ===========================================================

app = FastAPI(
    title="MeatVision AI",
    description="Production-grade AI application for meat species identification and freshness detection.",
    version="2.0.0",
)

# ===========================================================
# PATHS & DIRECTORIES
# ===========================================================

FRONTEND_DIR = PROJECT_ROOT / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"
TEMPLATE_DIR = FRONTEND_DIR / "templates"
UPLOAD_FOLDER = STATIC_DIR / "uploads"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# ===========================================================
# STATIC & TEMPLATES
# ===========================================================

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# ===========================================================
# ENDPOINTS
# ===========================================================


@app.get("/", response_class=HTMLResponse, summary="Dashboard Home")
async def home(request: Request):
    """Renders the main dashboard for image upload and server diagnostics."""
    return templates.TemplateResponse(
        request=request, name="index.html", context={"request": request}
    )


@app.get("/health", response_class=JSONResponse, summary="System Health Check")
async def health_check():
    """Production health check endpoint for monitoring system and model status."""
    return {
        "status": "healthy",
        "service": "MeatVision AI Engine",
        "version": "2.0.0",
        "compute_device": str(DEVICE),
        "models": {
            "species_model": "loaded",
            "species_classes": SPECIES_CLASSES,
            "freshness_model": "loaded",
            "freshness_classes": FRESHNESS_CLASSES,
        },
    }


@app.post("/predict", response_class=HTMLResponse, summary="Execute Model Inference")
async def predict_image(request: Request, image: UploadFile = File(...)):
    """Accepts an uploaded meat image file and executes species & freshness inference."""
    if not image or not image.filename:
        raise HTTPException(status_code=400, detail="No image file provided.")

    save_path = UPLOAD_FOLDER / image.filename

    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        result = predict(str(save_path))

        return templates.TemplateResponse(
            request=request,
            name="result.html",
            context={
                "request": request,
                "image_name": image.filename,
                "is_valid": result.get("is_valid", True),
                "error_message": result.get("error_message", ""),
                "detected_category": result.get("detected_category", ""),
                "species": result["species"],
                "species_confidence": result["species_confidence"],
                "freshness": result["freshness"],
                "freshness_confidence": result["freshness_confidence"],
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Inference processing failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
