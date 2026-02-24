"""
DRONE AI — Energy-Aware Flight Planner (Feature 7)
Reads battery directly from Pi's ADC and adapts the mission:
  - Shortens path when battery is low
  - Returns home early
  - Prioritizes nearest waypoints
"""

from dataclasses import dataclass
from enum import Enum, auto
from utils.logger import get_logger
from config import settings

log = get_logger("energy_planner")


class BatteryState(Enum):
    FULL = auto()       # > 80%
    GOOD = auto()       # 50-80%
    LOW = auto()        # 30-50%
    WARNING = auto()    # 20-30%
    CRITICAL = auto()   # < 20%


@dataclass
class BatteryInfo:
    voltage: float = 0.0
    percentage: float = 100.0
    state: BatteryState = BatteryState.FULL
    estimated_flight_time_s: float = 600.0


class EnergyPlanner:
    """
    Monitors battery state and advises the mission planner.
    """

    def __init__(self):
        self.battery = BatteryInfo()
        self._voltage_history: list[float] = []

    def update_voltage(self, voltage: float):
        """Update battery voltage (called from BatteryReader)."""
        self.battery.voltage = voltage
        self._voltage_history.append(voltage)

        # Percentage (linear for 3S LiPo)
        v_range = settings.BATTERY_FULL_VOLTAGE - settings.BATTERY_CRITICAL_VOLTAGE
        v_above_min = max(0, voltage - settings.BATTERY_CRITICAL_VOLTAGE)
        self.battery.percentage = min(100, (v_above_min / v_range) * 100)

        # State
        pct = self.battery.percentage
        if pct > 80:
            self.battery.state = BatteryState.FULL
        elif pct > 50:
            self.battery.state = BatteryState.GOOD
        elif pct > 30:
            self.battery.state = BatteryState.LOW
        elif pct > settings.RETURN_HOME_RESERVE_PCT:
            self.battery.state = BatteryState.WARNING
        else:
            self.battery.state = BatteryState.CRITICAL

        # Estimate remaining flight time
        if len(self._voltage_history) >= 10:
            rate = (self._voltage_history[-10] - voltage) / 10
            if rate > 0:
                remaining_v = voltage - settings.BATTERY_CRITICAL_VOLTAGE
                self.battery.estimated_flight_time_s = remaining_v / rate

    def should_return_home(self) -> bool:
        return self.battery.state in (BatteryState.WARNING, BatteryState.CRITICAL)

    def should_force_land(self) -> bool:
        return self.battery.state == BatteryState.CRITICAL

    def get_max_waypoints_remaining(self) -> int:
        time_per_wp = 15.0
        reserve_s = self.battery.estimated_flight_time_s * (settings.RETURN_HOME_RESERVE_PCT / 100)
        available_s = max(0, self.battery.estimated_flight_time_s - reserve_s)
        return max(0, int(available_s / time_per_wp))

    def prioritize_waypoints(self, waypoints: list, current_lat: float,
                             current_lon: float) -> list:
        """Re-order remaining waypoints by distance when battery is low."""
        if self.battery.state not in (BatteryState.LOW, BatteryState.WARNING):
            return waypoints

        from modules.mission_planner import MissionPlanner
        remaining = [wp for wp in waypoints if not wp.completed]
        max_wp = self.get_max_waypoints_remaining()

        remaining.sort(
            key=lambda wp: MissionPlanner.distance_m(
                current_lat, current_lon, wp.latitude, wp.longitude
            )
        )

        truncated = remaining[:max_wp]
        if len(truncated) < len(remaining):
            log.warning(
                f"Battery LOW — trimmed mission from {len(remaining)} "
                f"to {len(truncated)} waypoints"
            )
        return truncated

    def get_status_string(self) -> str:
        return (
            f"Battery: {self.battery.voltage:.1f}V "
            f"({self.battery.percentage:.0f}%) "
            f"[{self.battery.state.name}] "
            f"~{self.battery.estimated_flight_time_s:.0f}s remaining"
        )
