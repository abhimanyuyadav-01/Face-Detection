import numpy as np
import pytest
from PIL import Image

from backend.face_detection import FaceDetectionResult, FaceDetector


class _FakeCascade:
    """Stand-in for cv2.CascadeClassifier that returns a fixed detection.

    cv2's CascadeClassifier is a C extension object whose bound methods
    can't be monkeypatched directly, so tests that need a deterministic
    detection swap out the whole `.detector` attribute instead.
    """

    def __init__(self, boxes):
        self._boxes = np.array(boxes) if boxes else np.empty((0, 4), dtype=int)

    def detectMultiScale(self, gray, scaleFactor, minNeighbors, minSize):
        return self._boxes


@pytest.fixture(scope="module")
def detector() -> FaceDetector:
    return FaceDetector()


def test_detector_loads(detector):
    assert detector.detector is not None
    assert not detector.detector.empty()


def test_detect_on_blank_image_finds_nothing(detector):
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    result = detector.detect(blank)
    assert isinstance(result, FaceDetectionResult)
    assert result.count == 0
    assert result.boxes == []
    assert result.annotated_image.shape == blank.shape


@pytest.mark.parametrize(
    "image_factory",
    [
        lambda: np.full((60, 60, 3), 128, dtype=np.uint8),          # RGB ndarray
        lambda: np.full((60, 60), 128, dtype=np.uint8),              # grayscale ndarray
        lambda: np.full((60, 60, 4), 128, dtype=np.uint8),           # RGBA ndarray
        lambda: Image.fromarray(np.full((60, 60, 3), 128, dtype=np.uint8)),  # PIL Image
    ],
    ids=["rgb_array", "grayscale_array", "rgba_array", "pil_image"],
)
def test_detect_accepts_all_supported_input_types(detector, image_factory):
    result = detector.detect(image_factory())
    assert result.annotated_image.shape[2] == 3  # always normalized to RGB
    assert result.annotated_image.dtype == np.uint8


def test_detect_draws_true_red_boxes_in_rgb():
    detector = FaceDetector()
    detector.detector = _FakeCascade(boxes=[[10, 10, 40, 40]])

    image = np.full((100, 100, 3), 200, dtype=np.uint8)
    result = detector.detect(image, box_color=(255, 0, 0))

    assert result.count == 1
    assert result.boxes == [(10, 10, 40, 40)]
    # The top-left corner of the box should be pure red in RGB order —
    # this guards against an OpenCV BGR/RGB channel mix-up regression.
    assert tuple(result.annotated_image[10, 10]) == (255, 0, 0)


def test_detect_multiple_faces():
    detector = FaceDetector()
    detector.detector = _FakeCascade(boxes=[[5, 5, 20, 20], [50, 50, 20, 20]])

    image = np.full((100, 100, 3), 200, dtype=np.uint8)
    result = detector.detect(image)

    assert result.count == 2
    assert len(result.boxes) == 2
