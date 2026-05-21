FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt huggingface_hub

COPY backend/ ./backend/
COPY src/ ./src/

# Download model from HF Hub at build time
RUN mkdir -p /app/saved_models && \
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='aaditn/emotion-recognition-model', filename='emotion_model.pth', local_dir='/app/saved_models/')"

WORKDIR /app/backend

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
