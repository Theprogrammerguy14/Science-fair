"""
DRONE AI — Mission Planner Module
Generates autonomous search grid patterns and manages waypoints.
"""

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from utils.logger import get_logger
from config import settings

log = get_logger("mission_planner")


class FlightMode(Enum):
    MANUAL = auto()
    ASSISTED = auto()
    AUTONOMOUS = auto()
    RETURN_HOME = auto()
    EMERGENCY_LAND = auto()


@dataclass
class Waypoint:
    latitude: float
    longitude: float
    altitude: float = settings.DEFAULT_ALTITUDE_M
    action: str = "FLY"  # FLY, HOVER, SCAN, LAND
    completed: bool = False


@dataclass
class MissionState:
    mode: FlightMode = FlightMode.MANUAL
    waypoints: list = field(default_factory=list)
    current_wp_index: int = 0
    home_lat: float = 0.0
    home_lon: float = 0.0
    humans_found: list = field(default_factory=list)
    mission_active: bool = False
    area_covered_pct: float = 0.0


class MissionPlanner:
    """
    Plans and manages autonomous search missions.
    Generates lawnmower / grid patterns for area coverage.
    """

    def __init__(self):
        self.state = MissionState()

    # ── Grid Generation ──────────────────────────
    def generate_search_grid(self, center_lat: float, center_lon: float,
                             rows: int = None, cols: int = None,
                             cell_size_m: float = None) -> list[Waypoint]:
        """
        Generate a lawnmower (boustrophedon) search pattern.
        
        Args:
            center_lat/lon: Center of the search area
            rows/cols:      Grid dimensions
            cell_size_m:    Size of each grid cell in meters
        """
        rows = rows or settings.SEARCH_GRID_ROWS
        cols = cols or settings.SEARCH_GRID_COLS
        cell_size = cell_size_m or settings.GRID_CELL_SIZE_M

        waypoints = []
        for r in range(rows):
            col_range = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
            for c in col_range:
                lat, lon = self._offset_gps(
                    center_lat, center_lon,
                    north_m=(r - rows / 2) * cell_size,
                    east_m=(c - cols / 2) * cell_size,
                )
                wp = Waypoint(
                    latitude=lat,
                    longitude=lon,
                    altitude=settings.DEFAULT_ALTITUDE_M,
                    action="SCAN",
                )
                waypoints.append(wp)

        # Final waypoint: return to start
        waypoints.append(Waypoint(
            latitude=center_lat,
            longitude=center_lon,
            action="LAND",
        ))

        log.info(f"Generated search grid: {rows}×{cols} = {len(waypoints)} waypoints")
        return waypoints

    # ── Mission Control ──────────────────────────
    def start_mission(self, home_lat: float, home_lon: float):
        """Initialize a new autonomous mission."""
        self.state.home_lat = home_lat
        self.state.home_lon = home_lon
        self.state.waypoints = self.generate_search_grid(home_lat, home_lon)
        self.state.current_wp_index = 0
        self.state.mode = FlightMode.AUTONOMOUS
        self.state.mission_active = True
        self.state.humans_found = []
        log.info("Mission started — AUTONOMOUS mode")

    def get_current_waypoint(self) -> Waypoint | None:
        """Get the current target waypoint."""
        if self.state.current_wp_index < len(self.state.waypoints):
            return self.state.waypoints[self.state.current_wp_index]
        return None

    def advance_waypoint(self):
        """Mark current waypoint as done and move to next."""
        if self.state.current_wp_index < len(self.state.waypoints):
            self.state.waypoints[self.state.current_wp_index].completed = True
            self.state.current_wp_index += 1
            completed = sum(1 for wp in self.state.waypoints if wp.completed)
            self.state.area_covered_pct = completed / len(self.state.waypoints) * 100
            log.info(
                f"Waypoint {self.state.current_wp_index}/{len(self.state.waypoints)} "
                f"— {self.state.area_covered_pct:.0f}% covered"
            )

    def mark_human_found(self, lat: float, lon: float, confidence: float):
        """Record a human detection location."""
        self.state.humans_found.append({
            "lat": lat, "lon": lon,
            "confidence": confidence,
            "wp_index": self.state.current_wp_index,
        })
        log.info(f"HUMAN FOUND at ({lat:.6f}, {lon:.6f}) conf={confidence:.0%}")

    def trigger_return_home(self):
        """Switch to Return-To-Home mode."""
        self.state.mode = FlightMode.RETURN_HOME
        rth_wp = Waypoint(
            latitude=self.state.home_lat,
            longitude=self.state.home_lon,
            action="LAND",
        )
        self.state.waypoints = [rth_wp]
        self.state.current_wp_index = 0
        log.warning("RETURN TO HOME triggered")

    def trigger_emergency_land(self):
        """Immediate landing at current position."""
        self.state.mode = FlightMode.EMERGENCY_LAND
        log.critical("EMERGENCY LAND triggered")

    @property
    def is_mission_complete(self) -> bool:
        return self.state.current_wp_index >= len(self.state.waypoints)

    # ── GPS Utilities ────────────────────────────
    @staticmethod
    def _offset_gps(lat: float, lon: float, north_m: float, east_m: float):
        """Offset a GPS coordinate by meters (simple flat-earth approx)."""
        lat_offset = north_m / 111_111.0
        lon_offset = east_m / (111_111.0 * math.cos(math.radians(lat)))
        return lat + lat_offset, lon + lon_offset

    @staticmethod
    def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine distance in meters between two GPS points."""
        R = 6_371_000
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
