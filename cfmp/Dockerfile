ARG BUILD_FROM=ghcr.io/home-assistant/amd64-base-debian:bookworm
# hadolint ignore=DL3006
FROM ${BUILD_FROM}

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# libgomp1 is LightGBM's only system-level dependency (OpenMP). Debian
# bookworm is PEP 668 protected, hence the virtualenv.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        libgomp1 \
        ca-certificates; \
    python3 -m venv "${VIRTUAL_ENV}"; \
    pip install --upgrade pip setuptools wheel; \
    apt-get purge -y --auto-remove python3-venv; \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --only-binary=:all: -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt \
    && find /opt/venv -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && find /opt/venv -name 'tests' -type d -prune -exec rm -rf {} +

# Note /opt/app/app: the package sits one level down so that
# `python -m app.server` resolves from the /opt/app working directory.
COPY app /opt/app/app
COPY rootfs /

# Import at build time so a broken shared library fails the build rather
# than the first request.
RUN python -c "import lightgbm, numpy, pandas; print('lightgbm', lightgbm.__version__)"

WORKDIR /opt/app

ARG BUILD_ARCH
ARG BUILD_VERSION
LABEL io.hass.version="${BUILD_VERSION}" \
      io.hass.type="addon" \
      io.hass.arch="${BUILD_ARCH}"
