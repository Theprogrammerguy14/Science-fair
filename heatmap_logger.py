"""
DRONE AI — Heatmap Logger (Feature 5)
Records the entire mission for post-flight analysis:
  - Flight path replay
  - Detection heatmap
  - Time-based analysis

Generates visual outputs for the ground station dashboard.
"""

import os
import json
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict
from utils.logger import get_logger
from config import settings

log = get_logger("heatmap_logger")


@dataclass
class FlightPoint:
    timestamp: float
    latitude: float
    longitude: float
    altitude: float
    mode: str


@dataclass
class DetectionEvent:
    timestamp: float
    latitude: float
    longitude: float
    class_name: str
    confidence: float
    explanation: str = ""


@dataclass
class MissionLog:
    mission_id: str = ""
    start_time: str = ""
    end_time: str = ""
    flight_path: list = field(default_factory=list)
    detections: list = field(default_factory=list)
    total_distance_m: float = 0.0
    area_covered_pct: float = 0.0
    duration_s: float = 0.0


class HeatmapLogger:
    """
    Logs all mission data and generates visual outputs.
    """

    def __init__(self):
        self._log = MissionLog()
        self._start_time = 0.0
        self._prev_lat = 0.0
        self._prev_lon = 0.0
        self._recording = False

    # ── Recording ────────────────────────────────
    def start_recording(self, mission_id: str = ""):
        """Begin recording a new mission."""
        self._start_time = time.time()
        self._log = MissionLog(
            mission_id=mission_id or datetime.now().strftime("MISSION_%Y%m%d_%H%M%S"),
            start_time=datetime.now().isoformat(),
        )
        self._recording = True
        log.info(f"Recording started: {self._log.mission_id}")

    def stop_recording(self):
        """Stop recording and finalize the log."""
        if not self._recording:
            return
        self._log.end_time = datetime.now().isoformat()
        self._log.duration_s = time.time() - self._start_time
        self._recording = False
        log.info(
            f"Recording stopped: {self._log.duration_s:.0f}s, "
            f"{len(self._log.detections)} detections, "
            f"{self._log.total_distance_m:.0f}m traveled"
        )

    def log_position(self, lat: float, lon: float, alt: float, mode: str):
        """Log a flight path point (call periodically, e.g. every 1s)."""
        if not self._recording:
            return

        point = FlightPoint(
            timestamp=time.time(),
            latitude=lat,
            longitude=lon,
            altitude=alt,
            mode=mode,
        )
        self._log.flight_path.append(asdict(point))

        # Track distance
        if self._prev_lat != 0:
            from modules.mission_planner import MissionPlanner
            dist = MissionPlanner.distance_m(self._prev_lat, self._prev_lon, lat, lon)
            self._log.total_distance_m += dist
        self._prev_lat = lat
        self._prev_lon = lon

    def log_detection(self, lat: float, lon: float, class_name: str,
                      confidence: float, explanation: str = ""):
        """Log a detection event."""
        if not self._recording:
            return

        event = DetectionEvent(
            timestamp=time.time(),
            latitude=lat,
            longitude=lon,
            class_name=class_name,
            confidence=confidence,
            explanation=explanation,
        )
        self._log.detections.append(asdict(event))

    def set_area_covered(self, pct: float):
        self._log.area_covered_pct = pct

    # ── Export ────────────────────────────────────
    def save_json(self) -> str:
        """Save the mission log as JSON. Returns the file path."""
        os.makedirs(settings.LOG_DIR, exist_ok=True)
        filepath = os.path.join(settings.LOG_DIR, f"{self._log.mission_id}.json")
        with open(filepath, "w") as f:
            json.dump(asdict(self._log), f, indent=2)
        log.info(f"Mission log saved: {filepath}")
        return filepath

    def generate_heatmap(self) -> str | None:
        """
        Generate a detection heatmap image.
        Returns the file path to the generated PNG.
        """
        if not self._log.detections:
            log.info("No detections to plot")
            return None

        try:
            import matplotlib
            matplotlib.use("Agg")  # headless
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            log.warning("matplotlib not installed — cannot generate heatmap")
            return None

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Plot flight path
        if self._log.flight_path:
            path_lats = [p["latitude"] for p in self._log.flight_path]
            path_lons = [p["longitude"] for p in self._log.flight_path]
            ax.plot(path_lons, path_lats, "b-", alpha=0.4, linewidth=1, label="Flight path")
            ax.plot(path_lons[0], path_lats[0], "gs", markersize=10, label="Start")
            ax.plot(path_lons[-1], path_lats[-1], "r^", markersize=10, label="End")

        # Plot detection heatmap
        det_lats = [d["latitude"] for d in self._log.detections]
        det_lons = [d["longitude"] for d in self._log.detections]
        det_conf = [d["confidence"] for d in self._log.detections]

        scatter = ax.scatter(
            det_lons, det_lats,
            c=det_conf, cmap="YlOrRd", s=200, alpha=0.8,
            edgecolors="black", linewidths=1,
            vmin=0.5, vmax=1.0, zorder=5,
        )
        plt.colorbar(scatter, ax=ax, label="Detection Confidence")

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Mission: {self._log.mission_id}\n"
                     f"{len(self._log.detections)} detections | "
                     f"{self._log.total_distance_m:.0f}m traveled | "
                     f"{self._log.duration_s:.0f}s duration")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        os.makedirs(settings.LOG_DIR, exist_ok=True)
        filepath = os.path.join(settings.LOG_DIR, f"{self._log.mission_id}_heatmap.png")
        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()
        log.info(f"Heatmap saved: {filepath}")
        return filepath

    # ── Getters ──────────────────────────────────
    @property
    def mission_data(self) -> dict:
        return asdict(self._log)

    @property
    def detection_count(self) -> int:
        return len(self._log.detections)
