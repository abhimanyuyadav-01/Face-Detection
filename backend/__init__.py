"""
AI Playground — backend package.

This package contains the core, framework-agnostic logic used by the
face detection demo. None of these modules import Streamlit — they can be
used from a script, a notebook, a test suite, or any UI layer. The
`pages/` folder at the project root is the Streamlit layer that calls into
this package.

    backend/
        face_detection.py        Project 3 — OpenCV Haar Cascade
        sample_assets.py         Shared helper for fallback demo images
"""

from .face_detection import FaceDetector, FaceDetectionResult
from .sample_assets import fetch_sample_image

__all__ = [
    "FaceDetector",
    "FaceDetectionResult",
    "fetch_sample_image",
]

__version__ = "1.0.0"
