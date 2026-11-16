FROM docker.m.daocloud.io/python:3.11-slim

ARG COMPUTE_MODE=cpu
ENV COMPUTE_MODE=${COMPUTE_MODE} \
    PYTHONUNBUFFERED=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

WORKDIR /app

# 1. 替换为中国大陆 apt 镜像源（兼容 Debian 12 debian.sources 与传统 sources.list）
RUN if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources && \
        sed -i 's/security.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources; \
    elif [ -f /etc/apt/sources.list ]; then \
        sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
        sed -i 's/security.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list; \
    fi && \
    apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

# 2. 使用国内 PyTorch 镜像源 + 清华 PyPI 源按需安装 CPU 或 GPU 依赖
RUN if [ "$COMPUTE_MODE" = "gpu" ]; then \
        pip install --no-cache-dir torch torchvision \
            --index-url https://mirrors.aliyun.com/pytorch-wheels/cu124 \
            --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
        pip install --no-cache-dir onnxruntime-gpu; \
    else \
        pip install --no-cache-dir torch torchvision \
            --index-url https://mirrors.aliyun.com/pytorch-wheels/cpu \
            --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
        pip install --no-cache-dir onnxruntime openvino; \
    fi

# 3. 安装业务依赖（自动走环境变量配置的清华 PyPI 源）
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
