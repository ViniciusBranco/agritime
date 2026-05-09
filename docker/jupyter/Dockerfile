FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# System deps for geo, raster, prophet (cmdstan), lightgbm
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
        libpq-dev \
        libgdal-dev \
        gdal-bin \
        libgeos-dev \
        libproj-dev \
        libspatialindex-dev \
        libomp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python deps first for layer caching
COPY pyproject.toml ./
COPY src ./src
RUN pip install -e ".[dev]"

EXPOSE 8888 8000

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
