"""
DRONE AI — AI Detector Module
Runs TFLite (MobileNet-SSD / YOLOv5-Nano) inference on each frame.
Detects humans and obstacles, returns bounding boxes + confidence.
"""

import time
import numpy as np
from dataclasses import dataclass
from utils.logger import get_logger
from config import settings

log = get_logger("ai_detector")

# Try multiple TFLite backends — one of these will work
Interpreter = None

# Option 1: ai-edge-litert (Google's new name, works on Python 3.12+)
try:
    from ai_edge_litert.interpreter import Interpreter
    log.info("Using ai-edge-litert")
except ImportError:
    pass

# Option 2: tflite-runtime (classic, works on Python ≤3.11)
if Interpreter is None:
    try:
        from tflite_runtime.interpreter import Interpreter
        log.info("Using tflite-runtime")
    except ImportError:
        pass

# Option 3: Full TensorFlow (heavy but always works)
if Interpreter is None:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        log.info("Using tensorflow.lite")
    except ImportError:
        pass

if Interpreter is None:
    log.error(
        "No TFLite runtime found! Install one of:\n"
        "  pip install ai-edge-litert      (recommended for Python 3.12+)\n"
        "  pip install tflite-runtime       (for Python ≤3.11)\n"
        "  pip install tensorflow           (heavy fallback)"
    )


@dataclass
class Detection:
    """Single detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple  # (ymin, xmin, ymax, xmax) normalised 0-1
    pixel_bbox: tuple = (0, 0, 0, 0)  # (x1, y1, x2, y2) in original frame pixels
    is_human: bool = False


class AIDetector:
    """
    Loads a TFLite model and runs inference.
    Supports MobileNet-SSD (default) and can be swapped for YOLOv5-Nano.
    """

    def __init__(self):
        self._interpreter: Interpreter | None = None
        self._input_details = None
        self._output_details = None
        self._labels: list[str] = []
        self._inference_time_ms: float = 0.0

    # ── Setup ────────────────────────────────────
    def load_model(self) -> bool:
        """Load the TFLite model and label map."""
        if Interpreter is None:
            log.error("TFLite interpreter not available")
            return False

        try:
            self._interpreter = Interpreter(model_path=settings.MODEL_PATH)
            self._interpreter.allocate_tensors()
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            log.info(f"Model loaded: {settings.MODEL_PATH}")
            log.info(f"  Input shape : {self._input_details[0]['shape']}")
            log.info(f"  Input dtype : {self._input_details[0]['dtype']}")
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            return False

        # Load labels
        try:
            with open(settings.LABELS_PATH, "r") as f:
                self._labels = [line.strip() for line in f if line.strip()]
            log.info(f"Loaded {len(self._labels)} class labels")
        except FileNotFoundError:
            log.warning("Labels file not found — using numeric IDs")

        return True

    # ── Inference ────────────────────────────────
    def detect(self, rgb_frame: np.ndarray, frame_w: int = 640, frame_h: int = 480) -> list[Detection]:
        """
        Run inference on a preprocessed RGB frame.
        Returns a list of Detection objects above the confidence threshold.
        
        Args:
            rgb_frame: RGB image resized to model input size (e.g. 300×300)
            frame_w:   Original frame width (for pixel bbox conversion)
            frame_h:   Original frame height
        """
        if self._interpreter is None:
            return []

        # Prepare input tensor
        input_data = np.expand_dims(rgb_frame, axis=0)
        expected_dtype = self._input_details[0]["dtype"]
        if expected_dtype == np.uint8:
            input_data = input_data.astype(np.uint8)
        elif expected_dtype == np.float32:
            input_data = (input_data / 255.0).astype(np.float32)

        # Run inference
        t0 = time.time()
        self._interpreter.set_tensor(self._input_details[0]["index"], input_data)
        self._interpreter.invoke()
        self._inference_time_ms = (time.time() - t0) * 1000

        # Parse outputs (MobileNet-SSD format)
        # Output 0: bounding boxes  [1, N, 4]  (ymin, xmin, ymax, xmax)
        # Output 1: class IDs       [1, N]
        # Output 2: confidence      [1, N]
        # Output 3: num detections   [1]
        boxes = self._interpreter.get_tensor(self._output_details[0]["index"])[0]
        classes = self._interpreter.get_tensor(self._output_details[1]["index"])[0]
        scores = self._interpreter.get_tensor(self._output_details[2]["index"])[0]
        num_det = int(self._interpreter.get_tensor(self._output_details[3]["index"])[0])

        detections: list[Detection] = []
        for i in range(min(num_det, settings.MAX_DETECTIONS)):
            score = float(scores[i])
            if score < settings.DETECTION_THRESHOLD:
                continue

            class_id = int(classes[i])
            class_name = self._labels[class_id] if class_id < len(self._labels) else str(class_id)

            ymin, xmin, ymax, xmax = boxes[i]
            pixel_bbox = (
                int(xmin * frame_w),
                int(ymin * frame_h),
                int(xmax * frame_w),
                int(ymax * frame_h),
            )

            det = Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=score,
                bbox=(ymin, xmin, ymax, xmax),
                pixel_bbox=pixel_bbox,
                is_human=(class_id == settings.HUMAN_CLASS_ID),
            )
            detections.append(det)

        return detections

    # ── Helpers ───────────────────────────────────
    @property
    def inference_time_ms(self) -> float:
        return self._inference_time_ms

    def get_humans(self, detections: list[Detection]) -> list[Detection]:
        """Filter for human detections only."""
        return [d for d in detections if d.is_human]

    def draw_boxes(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw bounding boxes and labels on the frame (for display/streaming)."""
        import cv2
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.pixel_bbox
            color = (0, 0, 255) if det.is_human else (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{det.class_name} {det.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(
                annotated, label, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

        # Inference time overlay
        cv2.putText(
            annotated,
            f"Inference: {self._inference_time_ms:.0f} ms",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )
        return annotated
