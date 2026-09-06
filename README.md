<div align="center">
  
# ⚡ MeatVision AI
  
**Deep Learning Powered Meat Species Classification & Freshness Detection**

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI Pipeline](https://github.com/JoelJames889/MeatVisionAI/actions/workflows/ci.yml/badge.svg)](https://github.com/JoelJames889/MeatVisionAI/actions)

MeatVision AI is a production-grade deep learning application that automates meat quality inspection. It classifies meat **species** (Beef, Chicken, Fish, Pork) and assesses **freshness** (Fresh, Half-Fresh, Spoiled) from images — making it an invaluable tool for the food industry and automated quality assurance.

</div>

---

## 🌟 Key Features

- **Species Identification** — Accurately classifies Beef, Chicken, Fish, and Pork using EfficientNet B0 transfer learning (94–98% test accuracy).
- **Freshness Detection** — Assesses quality as Fresh, Half-Fresh, or Spoiled with calibrated confidence scores.
- **Premium Web Interface** — Glassmorphism-styled "AI Command Center" with animated SVG confidence rings, live terminal diagnostics, and drag-and-drop upload.
- **FastAPI Backend** — Lightning-fast inference API with health monitoring endpoint.
- **Dockerized** — One-command deployment with Docker Compose.

## 📂 Project Structure

```text
MeatVisionAI/
├── backend/            # FastAPI application, prediction logic, model architecture
│   ├── app.py          # FastAPI routes & server
│   ├── predictor.py    # Dual-model inference engine
│   ├── preprocess.py   # Image preprocessing pipeline
│   ├── config.py       # Path & model configuration
│   ├── utils.py        # Image loading utilities
│   └── models/         # Neural network architecture
├── frontend/           # Web UI templates & assets
│   ├── templates/      # Jinja2 HTML templates
│   └── static/         # CSS design system & uploads
├── models/             # Pre-trained PyTorch weights (.pth)
├── scripts/            # Training & data pipeline utilities
│   ├── training/       # Model training, evaluation, and prediction scripts
│   └── data_pipeline/  # Documentation of data engineering pipeline
├── notebooks/          # Colab GPU training notebooks
├── tests/              # Unit test suite
├── .github/            # CI/CD, issue & PR templates
├── Dockerfile          # Container configuration
├── docker-compose.yml  # Multi-service orchestration
├── pyproject.toml      # Python project metadata & tool config
└── requirements.txt    # Pinned Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- [Python 3.9+](https://www.python.org/)
- [Docker](https://www.docker.com/) (optional, for containerized setup)

### Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/JoelJames889/MeatVisionAI.git
   cd MeatVisionAI
   ```

2. **Create a virtual environment & install dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the FastAPI server:**
   ```bash
   uvicorn backend.app:app --reload
   ```

4. **Open your browser:**
   Navigate to `http://127.0.0.1:8000`

### 🐳 Docker Deployment

```bash
docker compose up --build
```

## 🧠 Model Architecture

| Model | Task | Backbone | Classes | Test Accuracy |
|-------|------|----------|---------|---------------|
| `species_model.pth` | Species Classification | EfficientNet B0 | Beef, Chicken, Fish, Pork | 94–98% |
| `freshness_model.pth` | Freshness Detection | EfficientNet B0 | Fresh, Half-Fresh, Spoiled | 94–98% |

Both models use **transfer learning** with ImageNet-pretrained weights and **dynamic class weighting** to handle imbalanced datasets.

## 🧪 Testing & Code Quality

```bash
# Run tests
pytest tests/ -v

# Lint
flake8 backend tests

# Format
black backend tests
```

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
