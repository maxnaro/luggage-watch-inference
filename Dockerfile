ARG DEEPSTREAM_IMAGE=nvcr.io/nvidia/deepstream:7.1-samples-multiarch
ARG CUDA_VER=12.6
ARG DS_YOLO_REPO=https://github.com/marcoslucianops/DeepStream-Yolo.git
ARG REID_MODEL_URL=https://api.ngc.nvidia.com/v2/models/nvidia/tao/reidentificationnet/versions/deployable_v1.0/files/resnet50_market1501.etlt

FROM ${DEEPSTREAM_IMAGE} AS parser-builder

ARG CUDA_VER
ARG DS_YOLO_REPO

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_VER=${CUDA_VER}

RUN set -eux; \
        apt-get update; \
        CUDA_PKG_VER="$(echo "${CUDA_VER}" | tr '.' '-')"; \
        apt-get install -y --no-install-recommends \
            git \
            build-essential \
            ca-certificates; \
        if [ ! -f "/usr/local/cuda-${CUDA_VER}/include/crt/host_defines.h" ]; then \
            apt-get install -y --no-install-recommends "cuda-cudart-dev-${CUDA_PKG_VER}" \
            || apt-get install -y --no-install-recommends "cuda-toolkit-${CUDA_PKG_VER}"; \
        fi; \
        if [ ! -x "/usr/local/cuda-${CUDA_VER}/bin/nvcc" ]; then \
            apt-get install -y --no-install-recommends "cuda-nvcc-${CUDA_PKG_VER}" \
            || apt-get install -y --no-install-recommends "cuda-compiler-${CUDA_PKG_VER}" \
            || apt-get install -y --no-install-recommends "cuda-toolkit-${CUDA_PKG_VER}"; \
        fi; \
        rm -rf /var/lib/apt/lists/*

# DeepStream-Yolo expects the unversioned include path on some installs.
RUN if [ ! -e /opt/nvidia/deepstream/deepstream ] && [ -d /opt/nvidia/deepstream/deepstream-7.1 ]; then \
      ln -s /opt/nvidia/deepstream/deepstream-7.1 /opt/nvidia/deepstream/deepstream; \
    fi

# Some DeepStream images place CUDA headers under targets/*/include.
RUN set -e; \
        CUDA_INCLUDE_DIR="/usr/local/cuda-${CUDA_VER}/include"; \
        if [ ! -f "${CUDA_INCLUDE_DIR}/crt/host_defines.h" ]; then \
            HOST_DEFINES_PATH="$(find /usr/local -path '*/include/crt/host_defines.h' | head -n 1)"; \
            if [ -n "${HOST_DEFINES_PATH}" ]; then \
                mkdir -p "${CUDA_INCLUDE_DIR}/crt"; \
                ln -sf "${HOST_DEFINES_PATH}" "${CUDA_INCLUDE_DIR}/crt/host_defines.h"; \
            fi; \
        fi

# Some CUDA packages ship libs under targets/<arch>/lib instead of lib64.
RUN set -e; \
        CUDA_LIB_DIR="/usr/local/cuda-${CUDA_VER}/lib64"; \
        if [ ! -f "${CUDA_LIB_DIR}/libcublas.so" ]; then \
            CUBLAS_PATH="$(find /usr/local /usr/lib -path '*/lib*/libcublas.so*' 2>/dev/null | head -n 1)"; \
            if [ -n "${CUBLAS_PATH}" ]; then \
                mkdir -p "${CUDA_LIB_DIR}"; \
                ln -sf "${CUBLAS_PATH}" "${CUDA_LIB_DIR}/libcublas.so"; \
            fi; \
        fi

RUN git clone --depth 1 ${DS_YOLO_REPO} /tmp/DeepStream-Yolo && \
    make -C /tmp/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo clean && \
    make -C /tmp/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo && \
    cp /tmp/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so /tmp/

FROM ${DEEPSTREAM_IMAGE}

ARG REID_MODEL_URL

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/inference

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-gi \
    python3-gst-1.0 \
    gstreamer1.0-libav \
    python3-dev \
    pkg-config \
    libcairo2-dev \
    libgirepository1.0-dev \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# The DeepStream base image ships package metadata for several FFmpeg/codec
# libraries but strips the actual .so files.  --reinstall forces apt to
# re-extract them so the GStreamer libav plugin can load.
RUN apt-get update && apt-get install --reinstall -y \
    libavcodec58 \
    libavutil56 \
    libswresample3 \
    libavformat58 \
    libmp3lame0 \
    libx264-163 \
    libx265-199 \
    libxvidcore4 \
    libvpx7 \
    libmpg123-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig

COPY source/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    grep -Ev "^(pyds|pyds-stubs|PyGObject|PyGObject-stubs)([[:space:]]|$|@|=)" /tmp/requirements.txt > /tmp/requirements-runtime.txt && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements-runtime.txt

RUN set -eux; \
        if python3 -m pip show pyds >/dev/null 2>&1; then \
            exit 0; \
        fi; \
        PYDS_WHEEL="$(find /opt/nvidia -type f -name 'pyds-*.whl' | head -n 1 || true)"; \
        if [ -n "${PYDS_WHEEL}" ]; then \
            python3 -m pip install --no-cache-dir "${PYDS_WHEEL}"; \
        fi; \
        if python3 -m pip show pyds >/dev/null 2>&1; then \
            exit 0; \
        fi; \
        PY_TAG="$(python3 -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"; \
        ARCH="$(uname -m)"; \
        if [ "${ARCH}" = "x86_64" ]; then \
            PLATFORM_TAG="linux_x86_64"; \
        elif [ "${ARCH}" = "aarch64" ]; then \
            PLATFORM_TAG="linux_aarch64"; \
        else \
            echo "Unsupported architecture for automatic pyds wheel download: ${ARCH}" >&2; \
            exit 1; \
        fi; \
        PYDS_URL="https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.0/pyds-1.2.0-${PY_TAG}-${PY_TAG}-${PLATFORM_TAG}.whl"; \
        python3 -m pip install --no-cache-dir "${PYDS_URL}"; \
        python3 -m pip show pyds >/dev/null

COPY . .

COPY --from=parser-builder /tmp/libnvdsinfer_custom_impl_Yolo.so /workspace/inference/config/libnvdsinfer_custom_impl_Yolo.so

RUN wget -q -O /workspace/inference/model/resnet50_market1501.etlt ${REID_MODEL_URL}

RUN ln -sf /workspace/inference/model/model_b1_gpu0_fp16.engine /workspace/inference/model_b1_gpu0_fp16.engine

ENV PYTHONPATH=/workspace/inference/source

ENTRYPOINT ["python3", "source/app.py"]
