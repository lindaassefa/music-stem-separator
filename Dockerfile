FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STEM_DEVICE=cpu \
    PORT=8080 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    python -m pip install \
      torch==2.4.1+cpu torchaudio==2.4.1+cpu \
      --index-url https://download.pytorch.org/whl/cpu && \
    grep -v -E '^(torch|torchaudio)==' requirements.txt > /tmp/requirements-cpu.txt && \
    python -m pip install -r /tmp/requirements-cpu.txt && \
    python -m pip install demucs==4.0.1 --no-deps

COPY app.py .
COPY stem_ai/ stem_ai/

EXPOSE 8080

CMD ["python", "app.py"]
