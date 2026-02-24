"""
DRONE AI — Camera Module
Captures frames from the Pi camera / USB webcam and preprocesses for AI.
"""

import threading
import time
import cv2
import numpy as np
from utils.logger import get_logger
from config import settings

log = get_logger("camera")


class Camera:
    """Threaded camera capture with frame buffer."""

    def __init__(self):
        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._fps_actual = 0.0

    # ── Lifecycle ────────────────────────────────
    def start(self) -> bool:
        """Open the camera and start the capture thread."""
        self._cap = cv2.VideoCapture(settings.CAMERA_INDEX)
        if not self._cap.isOpened():
            log.error("Cannot open camera")
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.CAMERA_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.CAMERA_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS, settings.CAMERA_FPS)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        log.info(
            f"Camera started ({settings.CAMERA_WIDTH}x{settings.CAMERA_HEIGHT} "
            f"@ {settings.CAMERA_FPS} fps)"
        )
        return True

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()
        log.info("Camera stopped")

    # ── Frame Access ─────────────────────────────
    def get_frame(self) -> np.ndarray | None:
        """Return the latest frame (BGR, full resolution). Thread-safe."""
        with self._lock:
            if self._frame is not None:
                return self._frame.copy()
        return None

    def get_frame_for_model(self) -> np.ndarray | None:
        """Return a resized, RGB frame ready for the TFLite model."""
        frame = self.get_frame()
        if frame is None:
            return None
        w, h = settings.INPUT_SIZE
        resized = cv2.resize(frame, (w, h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb

    def get_jpeg(self, quality: int = 70) -> bytes | None:
        """Return the latest frame as JPEG bytes (for streaming)."""
        frame = self.get_frame()
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None

    @property
    def fps(self) -> float:
        return self._fps_actual

    # ── Internal ─────────────────────────────────
    def _capture_loop(self):
        t0 = time.time()
        count = 0
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
                count += 1
                elapsed = time.time() - t0
                if elapsed >= 2.0:
                    self._fps_actual = count / elapsed
                    count = 0
                    t0 = time.time()
            else:
                time.sleep(0.01)
