"""
Shared helper for fetching small fallback/demo images.

Projects 2 (Image Recognition) and 3 (Face Detection) both want to work
out of the box, even before a user uploads their own photo — this mirrors
the original notebook's "upload your own, or we'll grab a sample" pattern.
Images are fetched into memory (never written to disk), so this works
cleanly on read-only or ephemeral deployment filesystems.

This module has no Streamlit dependency; the Streamlit pages wrap
`fetch_sample_image()` with `st.cache_data` so the download only happens
once per session.
"""

from __future__ import annotations

import urllib.request
from io import BytesIO

from PIL import Image

# Long-lived sample images from OpenCV's own official sample-data folder —
# the same source the original notebook used as its fallback.
SAMPLE_IMAGES = {
    "object": {
        "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg",
        "caption": "Sample photo (OpenCV sample dataset) — used when no photo is uploaded",
    },
    "face": {
        "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
        "caption": "Classic computer-vision test photo (OpenCV sample dataset)",
    },
}


def fetch_sample_image(key: str, timeout: int = 10) -> Image.Image:
    """Downloads a small fallback sample image and returns it as a PIL Image.

    Raises
    ------
    KeyError
        If `key` isn't one of the known sample images.
    URLError / TimeoutError
        If the image can't be downloaded (e.g. no internet access) — the
        caller (a Streamlit page) is expected to catch this and prompt the
        user to upload their own photo instead.
    """
    if key not in SAMPLE_IMAGES:
        raise KeyError(f"Unknown sample image key {key!r}. Choose from {list(SAMPLE_IMAGES)}.")

    url = SAMPLE_IMAGES[key]["url"]
    with urllib.request.urlopen(url, timeout=timeout) as response:
        raw_bytes = response.read()
    return Image.open(BytesIO(raw_bytes)).convert("RGB")
