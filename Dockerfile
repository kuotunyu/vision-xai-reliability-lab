# CPU-first image: torch resolves from the PyTorch CPU index via uv.lock,
# so no CUDA libraries end up in the image. No dataset, weights, or secrets.

# --- builder ---
FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.9.22 /uv /uvx /usr/local/bin/
WORKDIR /app
# UV_PYTHON_PREFERENCE=only-system overrides pyproject.toml's
# python-preference=only-managed (which exists to dodge a local-dev Anaconda
# DLL conflict, see FAILURES.md) — this container's base image already has a
# clean Python 3.11, and combining only-managed with PYTHON_DOWNLOADS=never
# would hard-fail the build ("No interpreter found ... downloads set to
# 'never'") since it would refuse that perfectly good system interpreter.
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON_PREFERENCE=only-system \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Dependency layer (cached until pyproject/uv.lock change)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --group serve --no-install-project

# Project layer (non-editable so the venv is self-contained)
COPY src/ src/
RUN uv sync --frozen --no-dev --group serve --no-editable

# --- runtime ---
FROM python:3.11-slim
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY configs/ configs/
COPY app/ app/
# The evidence workbench needs only committed, aggregate, weight-free artifacts.
# Keep these COPY instructions explicit so raw results can never enter by accident.
COPY results/derived/summary.json results/derived/summary.json
COPY release/cuda-resume-canary.json release/cuda-resume-canary.json
COPY assets/figures/ assets/figures/
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    GRADIO_ANALYTICS_ENABLED=False
USER appuser
EXPOSE 8000
# The API answers /health as soon as it is up; models load lazily on demand.
HEALTHCHECK --interval=60s --timeout=30s --start-period=30s CMD ["python", "-c", \
    "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).status==200 else 1)"]
CMD ["python", "-m", "vision_xai.cli", "serve", "--config", "configs/smoke.yaml", "--host", "0.0.0.0", "--port", "8000"]
