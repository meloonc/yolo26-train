FROM python:3.11-slim

ARG COMPUTE_MODE=cpu
ENV COMPUTE_MODE=${COMPUTE_MODE} \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

RUN if [ "$COMPUTE_MODE" = "gpu" ]; then \
        pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124 && \
        pip install --no-cache-dir onnxruntime-gpu; \
    else \
        pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
        pip install --no-cache-dir onnxruntime openvino; \
    fi

RUN pip install --no-cache-dir \
    ultralytics>=8.4.0 \
    fastapi \
    uvicorn \
    python-multipart \
    pillow \
    pyyaml \
    onnx>=1.15.0 \
    streamlit \
    requests

COPY . /app
EXPOSE 8000 8501

CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port 8000 & streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0"
