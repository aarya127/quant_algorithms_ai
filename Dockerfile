FROM python:3.11-slim

WORKDIR /app

# Copy and install Python deps first (layer caching)
COPY backend/requirements.txt .

# Install CPU-only PyTorch (~180 MB) instead of the default CUDA wheel (~2.5 GB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

WORKDIR /app/backend

EXPOSE 8080

CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-8080} --workers 1 --timeout 120"]
