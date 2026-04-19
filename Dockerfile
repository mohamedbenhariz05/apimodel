# ─── Stage 1: dependency builder ───────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /install

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install/deps --no-cache-dir \
        flask==3.0.0 \
        gunicorn==22.0.0 \
        pillow==10.4.0 && \
    pip install --prefix=/install/deps --no-cache-dir \
        torch==2.3.1+cpu \
        torchvision==0.18.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu

# ─── Stage 2: lean runtime image ────────────────────────────────────────────
FROM python:3.11-slim

# Non-root user for security
RUN useradd --create-home appuser
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install/deps /usr/local

# Copy application code and model weights
COPY app.py .
COPY student_best.pth .

# Torch CPU-only: disable CUDA env noise
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=2

USER appuser
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"

# 2 workers × 2 threads — tune to your CPU core count
CMD ["gunicorn", "app:app", \
     "--workers", "2", \
     "--threads", "2", \
     "--bind", "0.0.0.0:5000", \
     "--timeout", "60", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
