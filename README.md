<div align="center">
  
# ⚡ MeatVision AI
  
**Production Deep Learning System for Optical Meat Species Identification & Freshness Grading**

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI Pipeline](https://github.com/JoelJames889/MeatVisionAI/actions/workflows/ci.yml/badge.svg)](https://github.com/JoelJames889/MeatVisionAI/actions)

MeatVision AI is a production-grade automated computer vision platform designed for industrial slaughterhouses, automated butcher facilities, and food quality kiosks. It concurrently classifies **meat species** (Beef, Chicken, Fish, Pork) and grades **freshness quality** (Grade A - Fresh, Grade B - Half-Fresh, Grade C - Spoiled) using calibrated transfer-learning neural networks, backed by a defensive 4-tier Out-of-Distribution (OOD) validator that automatically rejects non-meat images.

</div>

---

## 🌟 Key Features

- **Decoupled Dual-Stream Inference** — Specialized EfficientNet-B0 architectures separate structural species features (muscle density, marbling) from chemical oxidation freshness features (surface discoloration, sheen).
- **Species Identification** — High-precision classification across **Beef**, **Chicken**, **Fish**, and **Pork** (100% test accuracy on balanced subsets).
- **Freshness Quality Grading** — Multi-class grading across **Grade A (Peak Freshness)**, **Grade B (Marginal / Cook Promptly)**, and **Grade C (Spoilage Confirmed)** with calibrated softmax probability distributions.
- **4-Tier Defensive OOD Gatekeeper** — Automatic rejection of random non-meat images (domestic dogs/cats, vehicles, electronics, furniture, apparel, documents, landscapes, abstract noise) using MobileNet-V3 semantic classification and HSV flesh chrominance analysis.
- **Glassmorphism Command Center** — Modern responsive UI with animated SVG circular confidence dials, horizontal probability distribution bars, and biological inspection profile cards.
- **Dual API Access** — Supports server-side rendered HTML results and programmatic JSON inference (`/api/predict`).
- **Containerized Deployment** — Pre-configured Docker and Docker Compose workflows.

---

## 📂 Project Architecture

```text
MeatVisionAI/
├── backend/
│   ├── app.py              # FastAPI server, HTML & JSON REST endpoints
│   ├── predictor.py        # Dual-stream inference engine & dynamic hot-reloader
│   ├── validator.py        # 4-tier OOD & synthetic image gatekeeper
│   ├── preprocess.py       # Input standardization & ImageNet normalization
│   ├── config.py           # Model paths, image resolution & device setup
│   ├── utils.py            # RGB image loading & bitstream verification
│   └── models/
│       └── architecture.py # MeatVisionModel wrapper (EfficientNet-B0 & ConvNeXt-Tiny)
├── frontend/
│   ├── templates/
│   │   ├── index.html      # Operator dashboard & file upload interface
│   │   └── result.html     # Glassmorphism metrics HUD & probability breakdown
│   └── static/
│       ├── css/style.css   # Dark-mode HSL design tokens & animations
│       └── uploads/        # Ingested temporary samples
├── models/
│   ├── species_model.pth   # Trained species classifier weights (16.35 MB)
│   └── freshness_model.pth # Calibrated freshness grading weights (16.34 MB)
├── scripts/
│   ├── training/
│   │   ├── config.py       # Hyperparameters & multi-environment dataset resolution
│   │   ├── dataset.py      # Zero-copy TransformSubset loader & class balancer
│   │   ├── trainer.py      # Fit loop, early stopping & checkpoint manager
│   │   ├── evaluate.py     # Multi-class precision, recall, F1, confusion matrix
│   │   ├── train_species.py
│   │   ├── train_freshness.py
│   │   └── utils.py        # Task-specific data augmentation policies
│   └── data_pipeline/      # Data engineering documentation
├── notebooks/              # Google Colab GPU training notebooks
├── tests/                  # 19 comprehensive unit & integration tests
├── Dockerfile              # Production container definition
├── docker-compose.yml      # Service orchestration
├── pyproject.toml          # Build configuration & tool metadata
└── requirements.txt        # Pinned dependencies
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+** (Tested on Python 3.10 and 3.11)
- **PyTorch 2.0+** & **Torchvision 0.15+**

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/JoelJames889/MeatVisionAI.git
   cd MeatVisionAI
   ```

2. **Create a virtual environment & install dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Launch the FastAPI application server:**
   ```bash
   uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
   ```

4. **Access the Web Dashboard:**
   Open your browser to: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 📡 API Endpoints

| Method | Endpoint | Description | Response Type |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Web operator dashboard with upload dropzone | HTML |
| `GET` | `/health` | System health check, device status, and loaded models | JSON |
| `POST` | `/predict` | Multipart file upload returning rendered analysis view | HTML |
| `POST` | `/api/predict` | Multipart file upload returning raw inference metrics | JSON |

### Sample JSON API Response (`/api/predict`)

```json
{
  "is_valid": true,
  "species": "Beef",
  "species_confidence": 99.85,
  "species_probabilities": {
    "Beef": 99.85,
    "Chicken": 0.08,
    "Fish": 0.04,
    "Pork": 0.03
  },
  "freshness": "Fresh",
  "freshness_confidence": 96.98,
  "freshness_probabilities": {
    "Fresh": 96.98,
    "Half Fresh": 2.15,
    "Spoiled": 0.87
  },
  "freshness_status": "Grade A - Peak Freshness",
  "freshness_description": "Safe for consumption. Preserved texture, natural pigmentation, and no microbial spoilage detected.",
  "freshness_color": "#10b981",
  "quality_guidance": "Inspection evaluates oxymyoglobin/metmyoglobin surface oxidation, fat marbling tint, and tissue elasticity."
}
```

---

## 🧠 Model Architectures & Benchmarks

| Stream | Model Weights | Neural Backbone | Classes | Validation Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **Species** | `species_model.pth` | EfficientNet-B0 | Beef, Chicken, Fish, Pork | 4-class Softmax with Weighted Class Sampler |
| **Freshness** | `freshness_model.pth` | EfficientNet-B0 | Fresh, Half Fresh, Spoiled | 3-class Softmax with AdamW & Cosine Annealing |
| **OOD Gate** | MobileNet-V3 Small | MobileNet-V3 | ImageNet-1K + HSV flesh mask | 4-tier rejection gatekeeper |

---

## 🧪 Testing & Code Quality

Run the automated test suite covering preprocessing, model architecture, OOD validation, and REST endpoints:

```bash
# Run complete test suite (19 tests)
pytest tests/ -v
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
