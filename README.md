# MeatVision

MeatVision is an AI-powered application designed to classify the species and freshness of meat from images. It uses advanced deep learning models to assess meat quality, making it a valuable tool for automated inspection.

## Features
- **Species Identification**: Identifies the species of meat (Beef, Chicken, Pork, Fish).
- **Freshness Detection**: Assesses whether the meat is fresh, half-fresh, or spoiled.
- **Web Interface**: A FastAPI and frontend integration that allows easy image uploads and real-time predictions.

## Repository Structure
- `Backend/`: Contains the FastAPI application, prediction logic, and API endpoints.
- `Frontend/`: Contains HTML/CSS templates for the user interface.
- `Models/`: Directory for storing pre-trained model weights (e.g., PyTorch `.pth` files).
- `Scripts/`: Contains utility scripts for training, data augmentation, and dataset balancing.
- `Dataset/`: The final organized dataset for training and validation.

## Getting Started

### Prerequisites
- Python 3.9+
- Docker (optional, for containerized deployment)

### Local Setup
1. Clone the repository and navigate to the root directory.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the FastAPI server:
   ```bash
   uvicorn Backend.app:app --reload
   ```
4. Access the web application at `http://127.0.0.1:8000`.

## Model Training
If you wish to train the models from scratch, refer to the scripts in the `Scripts/models/` directory. The dataset is already curated and available in the `Dataset/` directory.

## Testing
Run unit tests to verify the application functionality:
```bash
pytest tests/
```

## Code Quality
This project uses `flake8` for linting and `black` for code formatting.
```bash
flake8 Backend Scripts
black Backend Scripts
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
