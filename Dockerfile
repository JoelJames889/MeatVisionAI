FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (e.g. for OpenCV)
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy directories
COPY Backend/ ./Backend/
COPY Frontend/ ./Frontend/
COPY Models/ ./Models/

EXPOSE 8000

WORKDIR /app
CMD ["uvicorn", "Backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
