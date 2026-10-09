# Research papers are compiled in this stage so the runtime image carries only the
# PDFs, not a LaTeX install. A paper that fails to compile is skipped, not fatal.
FROM debian:bookworm-slim AS papers
RUN apt-get update && apt-get install -y --no-install-recommends \
        texlive-latex-base texlive-latex-recommended texlive-fonts-recommended \
    && rm -rf /var/lib/apt/lists/*
COPY algorithms /src/algorithms
WORKDIR /src
RUN find algorithms -name '*theory.tex' | while read -r f; do \
        (cd "$(dirname "$f")" && for _ in 1 2; do \
            pdflatex -interaction=nonstopmode "$(basename "$f")" > /dev/null; done) || true; \
    done; \
    mkdir /out && find algorithms -name '*theory.pdf' -exec cp --parents {} /out \; \
    && find /out -name '*.pdf'

# The diagnostics notebook, rendered to HTML (code hidden) the same way, so the
# runtime image needs no Jupyter.
FROM python:3.11-slim AS notebook
RUN pip install --no-cache-dir nbconvert
COPY algorithms/volatility_forecasting/research/diagnostics.ipynb /src/
RUN mkdir -p /out/algorithms/volatility_forecasting/research \
    && jupyter nbconvert --to html --no-input /src/diagnostics.ipynb \
       --output-dir /out/algorithms/volatility_forecasting/research

FROM python:3.11-slim

WORKDIR /app

# OpenMP runtime: LightGBM (and XGBoost) models from the registry won't load without it
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

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
COPY --from=papers /out/ /app/
COPY --from=notebook /out/ /app/

WORKDIR /app/backend

EXPOSE 8080

# Pre-flight: test that the app imports cleanly, surfacing any error in deploy logs.
COPY backend/entrypoint.sh .
RUN chmod +x entrypoint.sh

CMD ["./entrypoint.sh"]
