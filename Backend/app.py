import sys
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from Backend.predictor import predict

# ===========================================================
# APP
# ===========================================================

app = FastAPI(title="MeatVision AI")

# ===========================================================
# PATHS
# ===========================================================

PROJECT_ROOT = Path(r"D:\MeatVision_Project")

FRONTEND = PROJECT_ROOT / "Frontend"

STATIC_DIR = FRONTEND / "static"

TEMPLATE_DIR = FRONTEND / "templates"

UPLOAD_FOLDER = STATIC_DIR / "uploads"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# ===========================================================
# STATIC & TEMPLATES
# ===========================================================

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# ===========================================================
# HOME
# ===========================================================


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request, name="index.html", context={"request": request}
    )


# ===========================================================
# PREDICT
# ===========================================================


@app.post("/predict", response_class=HTMLResponse)
async def predict_image(request: Request, image: UploadFile = File(...)):

    save_path = UPLOAD_FOLDER / image.filename

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    result = predict(str(save_path))

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "request": request,
            "image_name": image.filename,
            "species": result["species"],
            "species_confidence": result["species_confidence"],
            "freshness": result["freshness"],
            "freshness_confidence": result["freshness_confidence"],
        },
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("Backend.app:app", host="0.0.0.0", port=8000, reload=True)
