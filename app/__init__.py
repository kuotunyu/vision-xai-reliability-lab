"""Serving layer (FastAPI + optional Gradio demo).

This package is intentionally not part of the installed wheel; it is imported
from the repository root (or the Docker WORKDIR), and its heavy dependencies
are loaded lazily.
"""
