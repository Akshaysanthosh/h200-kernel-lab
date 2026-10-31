FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    nvidia-cuda-runtime-cu12==12.8.90 \
    nvidia-cuda-nvcc-cu12==12.8.93 \
    nvidia-cublas-cu12==12.8.4.1
RUN pip install --no-cache-dir nvidia-cuda-cccl-cu12==12.8.90
WORKDIR /work
CMD ["python", "cuBLASLt/local-build/compile_probe.py"]
