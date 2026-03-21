FROM python:3.11-slim

WORKDIR /app

# Install system deps needed by torch/transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps first (layer caching)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

WORKDIR /app/backend

EXPOSE 8080

CMD ["python", "app.py"]
