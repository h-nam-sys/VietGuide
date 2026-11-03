# syntax=docker/dockerfile:1
FROM python:3.12-slim-trixie AS build
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=2
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build glslc libvulkan-dev git \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /opt/venv
COPY requirements.txt /tmp/requirements-native.txt
COPY deployment/container/build_requirements.py /tmp/build_requirements.py
COPY deployment/container/constraints.generated.txt /tmp/constraints.txt
COPY deployment/container/llama.generated.txt /tmp/llama.txt
RUN python /tmp/build_requirements.py /tmp/requirements-native.txt /tmp/requirements.txt
# Avoid downloading NVIDIA PyTorch libraries for the CPU retrieval runtime.
RUN pip install --no-cache-dir -c /tmp/constraints.txt torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -c /tmp/constraints.txt -r /tmp/requirements.txt
RUN apt-get update \
    && apt-get install -y --no-install-recommends spirv-headers glslang-tools \
    && rm -rf /var/lib/apt/lists/* \
    && CMAKE_ARGS="-DGGML_VULKAN=ON -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF" \
       pip install --no-cache-dir --no-binary=llama-cpp-python -r /tmp/llama.txt \
    && pip check

FROM python:3.12-slim-trixie
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/cache/huggingface \
    TOKENIZERS_PARALLELISM=false
RUN apt-get update && apt-get install -y --no-install-recommends \
    libvulkan1 mesa-vulkan-drivers libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && find /usr/share/vulkan/icd.d -iname "*lvp*" -delete
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY backend ./backend
COPY rag ./rag
COPY llm_service ./llm_service
COPY run_services.py qa_suite.py ./
COPY deployment/manage.py ./deployment/manage.py
COPY deployment/container ./deployment/container
RUN mkdir -p /app/data /app/logs /app/models /app/backend/static/forms /app/backups \
    && pip freeze > /app/deployment/container/runtime-requirements.txt
EXPOSE 8000 8501
ENTRYPOINT ["python", "/app/deployment/container/entrypoint.py"]
