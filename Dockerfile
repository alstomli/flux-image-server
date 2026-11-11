FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04@sha256:ac55d124da4882b497f732d8dfd9a702d5447a5f29d08d56da6f64f0a1eb34bc
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/workspace/huggingface FLUX_DATA_DIR=/opt/flux-runtime FLUX_MODEL_DIR=/workspace/models/flux2-dev
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /app
COPY pyproject.toml ./
COPY constraints-cpu.txt ./
COPY flux_image_server/ flux_image_server/
COPY scripts/ scripts/
RUN python -m pip install --no-cache-dir 'torch==2.9.1' 'torchvision==0.24.1' --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install --no-cache-dir -c constraints-cpu.txt '.[gpu]' \
    && python -m pip check
EXPOSE 8001 8889
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/healthz',timeout=3)" || exit 1
CMD ["python", "-m", "flux_image_server"]
