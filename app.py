"""Project 3 — Face Detection (Streamlit page).

Thin UI layer only. All the actual CV logic lives in
`backend/face_detection.py` and has its own unit tests in `tests/`.

Note: the original notebook captured webcam photos via a Colab-specific
JavaScript snippet. `st.camera_input()` is Streamlit's native, built-in
equivalent, so that hack isn't needed here at all.
"""
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.face_detection import FaceDetector  # noqa: E402
from backend.sample_assets import fetch_sample_image  # noqa: E402
from ui_helpers import inject_base_styles, render_footer, render_header  # noqa: E402

st.set_page_config(page_title="Face Detection · AI Playground", page_icon="👤", layout="wide")
inject_base_styles()

render_header(
    icon="👤",
    title="Face Detection",
    eyebrow="Project 3 · Computer Vision",
    subtitle="Locate every human face in a photo — this finds *where* faces are, not *whose* they are.",
)


@st.cache_resource(show_spinner="Loading the Haar Cascade face detector...")
def get_detector() -> FaceDetector:
    return FaceDetector()


@st.cache_data(show_spinner="Fetching a sample photo...")
def get_sample_image() -> Image.Image:
    return fetch_sample_image("face")


col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("#### 1. Choose a photo")
    input_mode = st.radio(
        "Image source",
        ["📤 Upload a photo", "📷 Use my camera"],
        horizontal=True,
        label_visibility="collapsed",
    )

    image, caption = None, None
    if input_mode == "📤 Upload a photo":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            caption = f"Your upload — {uploaded_file.name}"
    else:
        camera_file = st.camera_input("Take a photo")
        if camera_file is not None:
            image = Image.open(camera_file).convert("RGB")
            caption = "Captured from your camera"

    if image is None:
        try:
            image = get_sample_image()
            caption = "Sample photo — choose upload or camera above to try your own"
        except Exception as exc:
            st.error(
                "Couldn't fetch the sample photo (no internet access from this "
                f"server?). Please upload your own image instead.\n\nDetails: {exc}"
            )

    st.markdown("#### 2. Detection settings")
    scale_factor = st.slider(
        "Scale factor", min_value=1.05, max_value=1.5, value=1.1, step=0.05,
        help="How much the image shrinks at each scan step. Smaller = more thorough, slower.",
    )
    min_neighbors = st.slider(
        "Minimum neighbors", min_value=1, max_value=10, value=5,
        help="Higher = fewer false positives, but may miss real faces.",
    )

with col_result:
    st.markdown("#### 3. Detected faces")
    if image is not None:
        try:
            detector = get_detector()
            result = detector.detect(image, scale_factor=scale_factor, min_neighbors=min_neighbors)
            st.image(result.annotated_image, caption=caption, use_container_width=True)
            if result.count == 0:
                st.warning("No faces detected. Try a clearer, front-facing photo, or lower 'minimum neighbors'.")
            else:
                st.success(f"**{result.count}** face{'s' if result.count != 1 else ''} detected.")
        except Exception as exc:
            st.error(f"Something went wrong while running detection. Details: {exc}")
    else:
        st.info("Choose a photo on the left to see detected faces here.")

with st.expander("🧠 How it works"):
    st.markdown(
        """
**Pipeline:** `Image → Grayscale → Haar Cascade Scan → Face Coordinates → Boxes Drawn`

Face **detection** only answers *"is there a face here, and where?"* — it
is not face **recognition**, which would answer *"whose face is this?"*.

A Haar Cascade is a lightweight, pretrained detector that scans an image
at multiple scales looking for the contrast patterns typical of a human
face — for example, the eye region usually being darker than the bridge
of the nose. It works on a **grayscale** version of the image, since shape
and contrast matter more than color for this technique.
        """
    )

render_footer()
