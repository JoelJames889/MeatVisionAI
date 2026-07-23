<div align="center">
  
# 🥩 MeatVisionAI
  
**AI-Powered Meat Classification & Freshness Prediction**

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MeatVisionAI is an advanced deep learning application designed to automate the quality inspection of meat. It classifies meat species and assesses its freshness from images, making it an invaluable tool for the food industry and automated quality assurance.

</div>

---

## 🌟 Key Features

- **Species Identification**: Accurately identifies the species of meat (Beef, Chicken, Pork, Fish).
- **Freshness Detection**: Assesses the quality and categorizes the meat as **Fresh**, **Half-Fresh**, or **Spoiled**.
- **Intuitive Web Interface**: A sleek, user-friendly frontend integrated with a robust FastAPI backend for real-time predictions via image upload.
- **Dockerized**: Easy to deploy and run anywhere using Docker.

## 📂 Repository Structure

```text
MeatVisionAI/
├── Backend/        # FastAPI application, prediction logic, API endpoints
├── Frontend/       # HTML/CSS/JS templates for the web interface
├── Models/         # Pre-trained PyTorch model weights (.pth)
├── Scripts/        # Utility scripts (training, data augmentation, balancing)
├── Dataset/        # Organized datasets for model training & validation
├── tests/          # Unit tests for the application
├── Dockerfile      # Docker configuration for containerized deployment
└── requirements.txt # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- [Python 3.9+](https://www.python.org/)
- [Docker](https://www.docker.com/) (Optional, for containerized setup)

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
   uvicorn Backend.app:app --reload
   ```

4. **Access the application:**
   Open your browser and navigate to `http://127.0.0.1:8000`.

### 🐳 Docker Deployment

To build and run the application using Docker:

```bash
docker-compose up --build
```
*(Or use `docker build -t meatvision .` and `docker run -p 8000:8000 meatvision`)*

## 🧠 Model Training

If you wish to train the models from scratch or experiment with the architecture:
- Navigate to the `Scripts/models/` directory.
- The pre-processed dataset is available in the `Dataset/` directory.
- Run the training scripts provided to generate new `.pth` weights.

## 🧪 Testing & Code Quality

Run unit tests to verify functionality:
```bash
pytest tests/
```

This project adheres to strict coding standards using `flake8` and `black`:
```bash
flake8 Backend Scripts
black Backend Scripts
```

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
