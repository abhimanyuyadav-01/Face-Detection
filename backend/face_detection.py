"""
Project 3 — Face Detection (Computer Vision)

Locates human faces in a photo using OpenCV's pretrained Haar Cascade
classifier and draws a bounding box around each one.

Pipeline
--------
Image -> Grayscale -> Haar Cascade Scan -> Face Coordinates -> Boxes Drawn

Face *detection* only answers "is there a face here, and where?" — it is
not face *recognition*, which would answer "whose face is this?". Haar
cascades are a lightweight, pretrained detector that scans an image at
multiple scales looking for the contrast patterns typical of a human
face (e.g. the eyes being darker than the bridge of the nose). They
operate on grayscale images, since shape and contrast matter more than
color for this technique.

This module has no Streamlit (or any UI) dependency, so it can be reused
from a script, a notebook, a test suite, or a web app. It works directly
with Pillow images / NumPy arrays (in standard RGB order) rather than
requiring the caller to think about OpenCV's BGR channel order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from pathlib import Path
from typing import List, Tuple, Union

import cv2 as _cv2
import numpy as np
from PIL import Image


def _resolve_cv2_module():
    """Return the actual OpenCV extension module if cv2 is wrapped."""
    if hasattr(_cv2, "CascadeClassifier") and hasattr(_cv2, "data"):
        return _cv2

    for module_name in ("cv2.cv2",):
        try:
            nested_cv2 = importlib.import_module(module_name)
        except Exception:
            continue
        if hasattr(nested_cv2, "CascadeClassifier") and hasattr(nested_cv2, "data"):
            return nested_cv2

    return _cv2


cv2 = _resolve_cv2_module()

ImageLike = Union[str, Path, Image.Image, np.ndarray]


@dataclass
class FaceDetectionResult:
    """Result of running face detection on one image."""

    annotated_image: np.ndarray                    # RGB array, boxes drawn in
    boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)  # (x, y, w, h)

    @property
    def count(self) -> int:
        """Number of faces detected."""
        return len(self.boxes)


class FaceDetector:
    """Wraps an OpenCV Haar Cascade face detector.

    Example
    -------
    >>> detector = FaceDetector()
    >>> result = detector.detect("photo.jpg")
    >>> result.count
    2
    """

    def __init__(self, cascade_file: str = "haarcascade_frontalface_default.xml"):
        if not hasattr(cv2, "CascadeClassifier"):
            raise RuntimeError(
                "Your OpenCV installation does not expose CascadeClassifier. "
                "Reinstall opencv-python-headless and clear Streamlit's build cache."
            )

        cascade_path = cv2.data.haarcascades + cascade_file
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError(
                f"Could not load Haar Cascade file: {cascade_path}. "
                "Your OpenCV installation may be corrupted or missing its data files."
            )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_rgb_array(image: ImageLike) -> np.ndarray:
        """Normalizes any supported input into a standard RGB uint8 array."""
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        if isinstance(image, np.ndarray):
            if image.ndim == 2:  # already grayscale
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            if image.shape[2] == 4:  # has an alpha channel
                return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            return image  # assumed already RGB, 3 channels
        raise TypeError(f"Unsupported image type: {type(image)!r}")

    # ------------------------------------------------------------------ #
    # Detection
    # ------------------------------------------------------------------ #
    def detect(
        self,
        image: ImageLike,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30),
        box_color: Tuple[int, int, int] = (255, 0, 0),
        box_thickness: int = 3,
    ) -> FaceDetectionResult:
        """Detects faces and returns an annotated copy of the image plus their coordinates.

        Parameters
        ----------
        scale_factor:
            How much the image is shrunk at each scan step while searching
            for faces of different sizes. Smaller = more thorough, slower.
        min_neighbors:
            How many overlapping detections are required before a region
            is confidently marked as a face. Higher = fewer false
            positives, but may miss real faces.
        min_size:
            The smallest face (in pixels) the detector will consider.
        box_color:
            RGB color for the drawn bounding boxes (default: red).
        """
        rgb_image = self._to_rgb_array(image)
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
        )

        annotated = rgb_image.copy()
        boxes: List[Tuple[int, int, int, int]] = []
        for (x, y, w, h) in faces:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, box_thickness)
            boxes.append((int(x), int(y), int(w), int(h)))

        return FaceDetectionResult(annotated_image=annotated, boxes=boxes)


if __name__ == "__main__":
    detector = FaceDetector()
    print("Haar Cascade face detector loaded successfully.")

    # Smoke-test the full pipeline on a synthetic image so this script runs
    # anywhere, with no sample photo or internet connection required. A
    # blank/random image should correctly yield zero detections.
    blank_image = np.zeros((200, 200, 3), dtype=np.uint8)
    result = detector.detect(blank_image)
    print(f"Smoke test on a blank {blank_image.shape[:2]} image -> {result.count} faces detected (expected 0).")
