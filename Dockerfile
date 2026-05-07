FROM python:3.11-slim

WORKDIR /app

# Copy and install Python deps first (layer caching)
COPY backend/requirements.txt .

# Install CPU-only PyTorch (~180 MB) instead of the default CUDA wheel (~2.5 GB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download FinBERT weights at build time so the first request never times out
# doing a ~440 MB runtime download. The weights land in /root/.cache/huggingface/.
RUN python3 -c "\
from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
AutoTokenizer.from_pretrained('ProsusAI/finbert'); \
AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert'); \
print('FinBERT pre-downloaded')"

# Copy the rest of the project
COPY . .

WORKDIR /app/backend

EXPOSE 8080

# Pre-flight: test that the app imports cleanly, surfacing any error in deploy logs.
COPY backend/entrypoint.sh .
RUN chmod +x entrypoint.sh

CMD ["./entrypoint.sh"]
