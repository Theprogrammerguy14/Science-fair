"""
DRONE AI — Decision Engine (Pi-Only)
Core IF/THEN logic — runs every frame cycle.

Priority:
  1. Critical battery → force land
  2. Low battery → return home
  3. Human detected → alert, hover, signal
  4. Normal → continue mission
"""

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from modules.ai_detector import Detection
from modules.mission_planner import MissionPlanner, FlightMode
from modules.energy_planner import EnergyPlanner
from modules.explainable_ai import ExplainableAI, Explanation
from utils.logger import get_logger
from config import settings

log = get_logger("decision_engine")


class AlertLevel(Enum):
    NONE = auto()
    INFO = auto()
    DETECTION = auto()
    WARNING = auto()
    CRITICAL = auto()


@dataclass
class DecisionResult:
    alert_level: AlertLevel = AlertLevel.NONE
    flight_command: str = "HOLD"
    alert_message: str = ""
    hover_duration_s: float = 0.0
    explanations: list = field(default_factory=list)
    should_signal: bool = False


class DecisionEngine:
    """Evaluates all inputs each frame and produces a single DecisionResult."""

    def __init__(self, mission: MissionPlanner, energy: EnergyPlanner):
        self.mission = mission
        self.energy = energy
        self.xai = ExplainableAI()
        self._hover_until = 0.0
        self._detection_count = 0

    def evaluate(self, detections: list[Detection],
                 gps_lat: float, gps_lon: float,
                 frame=None, prev_frame=None) -> DecisionResult:
        """
        Main decision loop — called every frame.

        Priority:
          1. Critical battery → force land
          2. Low battery → return home
          3. Human detected → alert + hover
          4. Normal → continue mission
        """
        result = DecisionResult()
        now = time.time()

        # ── Priority 1: Critical Battery ─────────
        if self.energy.should_force_land():
            result.alert_level = AlertLevel.CRITICAL
            result.alert_message = "BATTERY CRITICAL — emergency landing"
            result.flight_command = "LAND"
            self.mission.trigger_emergency_land()
            log.critical("BATTERY CRITICAL — forced landing")
            return result

        # ── Priority 2: Low Battery ──────────────
        if self.energy.should_return_home():
            result.alert_level = AlertLevel.WARNING
            result.alert_message = f"Battery low — returning home. {self.energy.get_status_string()}"
            result.flight_command = "RTH"
            self.mission.trigger_return_home()
            log.warning("Battery low — RTH triggered")
            return result

        # ── Priority 3: Human Detection ──────────
        humans = [d for d in detections if d.is_human]
        if humans:
            self._detection_count += len(humans)
            best = max(humans, key=lambda d: d.confidence)

            explanations = self.xai.explain_all(humans, frame, prev_frame)
            result.explanations = explanations

            self.mission.mark_human_found(gps_lat, gps_lon, best.confidence)

            result.alert_level = AlertLevel.DETECTION
            result.flight_command = "HOVER"
            result.hover_duration_s = settings.HOVER_TIME_ON_DETECT_S
            result.should_signal = True
            result.alert_message = (
                f"HUMAN DETECTED — confidence {best.confidence:.0%} "
                f"at ({gps_lat:.6f}, {gps_lon:.6f})"
            )
            self._hover_until = now + settings.HOVER_TIME_ON_DETECT_S
            log.info(result.alert_message)
            for exp in explanations:
                log.info(f"  → {exp.summary}")
            return result

        # ── Priority 4: Currently Hovering ───────
        if now < self._hover_until:
            result.flight_command = "HOVER"
            result.alert_level = AlertLevel.INFO
            result.alert_message = f"Hovering — {self._hover_until - now:.1f}s remaining"
            return result

        # ── Priority 5: Normal Mission Flight ────
        if self.mission.state.mode == FlightMode.AUTONOMOUS:
            wp = self.mission.get_current_waypoint()
            if wp and not self.mission.is_mission_complete:
                dist = MissionPlanner.distance_m(gps_lat, gps_lon, wp.latitude, wp.longitude)
                if dist < 2.0:
                    self.mission.advance_waypoint()
                    wp = self.mission.get_current_waypoint()

                if wp:
                    result.flight_command = "FLY_TO_WP"
                    result.alert_message = (
                        f"Flying to WP {self.mission.state.current_wp_index + 1}"
                        f"/{len(self.mission.state.waypoints)} "
                        f"— {self.mission.state.area_covered_pct:.0f}% covered"
                    )
                else:
                    result.flight_command = "LAND"
                    result.alert_message = "Mission complete — landing"
            else:
                result.flight_command = "RTH"
                result.alert_message = "Mission complete — returning home"
                self.mission.trigger_return_home()
        elif self.mission.state.mode == FlightMode.RETURN_HOME:
            result.flight_command = "RTH"
            result.alert_message = "Returning home"
        else:
            result.flight_command = "HOLD"
            result.alert_message = "Manual mode — awaiting commands"

        return result

    @property
    def total_detections(self) -> int:
        return self._detection_count
