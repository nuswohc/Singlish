FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/user \
    NUMBA_CACHE_DIR=/home/user/.cache/numba \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

RUN useradd --create-home --uid 1000 user
WORKDIR /app
COPY mvp/requirements.txt /app/mvp/requirements.txt
RUN pip install --no-cache-dir -r /app/mvp/requirements.txt
COPY --chown=user:user mvp /app/mvp
USER user
RUN mkdir -p "$NUMBA_CACHE_DIR"

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "mvp.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "4"]
