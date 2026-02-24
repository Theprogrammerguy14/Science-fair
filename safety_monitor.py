"""
DRONE AI — Safety Monitor (Pi-Only)
Monitors all safety conditions directly on the Raspberry Pi:
  - Battery voltage (via ADC)
  - GPS fix quality
  - Dashboard connection (clients connected)
  - Kill switch state

If any critical condition fails → triggers failsafe.
"""

import time
import threading
from utils.logger import get_logger
from config import settings

log = get_logger("safety_monitor")


class SafetyMonitor:
    """
    Continuously checks all safety conditions.
    Any module can query is_safe_to_fly to know if flight is OK.
    """

    def __init__(self):
        self._gps_ok = False
        self._battery_ok = True
        self._battery_voltage = settings.BATTERY_FULL_VOLTAGE
        self._motors_ok = True
        self._failsafe_triggered = False
        self._dashboard_clients = 0
        self._lock = threading.Lock()

    # ── GPS Monitoring ───────────────────────────
    def update_gps(self, has_fix: bool, satellites: int = 0):
        with self._lock:
            self._gps_ok = has_fix and satellites >= 4
            if not self._gps_ok:
                log.debug(f"GPS: no fix (sats={satellites})")

    # ── Battery Monitoring ───────────────────────
    def update_battery(self, voltage: float):
        with self._lock:
            self._battery_voltage = voltage
            self._battery_ok = voltage > settings.BATTERY_CRITICAL_VOLTAGE
            if not self._battery_ok:
                log.warning(f"BATTERY CRITICAL: {voltage:.1f}V")
            elif voltage < settings.BATTERY_WARN_VOLTAGE:
                log.warning(f"Battery low: {voltage:.1f}V")

    # ── Dashboard Connection ─────────────────────
    def update_dashboard_clients(self, count: int):
        """Track how many clients are viewing the dashboard."""
        with self._lock:
            self._dashboard_clients = count

    @property
    def has_dashboard_client(self) -> bool:
        with self._lock:
            return self._dashboard_clients > 0

    # ── Motor Health ─────────────────────────────
    def update_motors(self, ok: bool):
        with self._lock:
            self._motors_ok = ok

    # ── Aggregate Safety Check ───────────────────
    @property
    def is_safe_to_fly(self) -> bool:
        """Master safety flag — battery and motors must be OK."""
        with self._lock:
            return self._battery_ok and self._motors_ok

    @property
    def gps_ok(self) -> bool:
        with self._lock:
            return self._gps_ok

    @property
    def battery_ok(self) -> bool:
        with self._lock:
            return self._battery_ok

    @property
    def battery_voltage(self) -> float:
        with self._lock:
            return self._battery_voltage

    @property
    def failsafe_triggered(self) -> bool:
        with self._lock:
            return self._failsafe_triggered

    def trigger_failsafe(self, reason: str = ""):
        with self._lock:
            self._failsafe_triggered = True
        log.critical(f"FAILSAFE TRIGGERED: {reason}")

    def reset_failsafe(self):
        with self._lock:
            if self._battery_ok:
                self._failsafe_triggered = False
                log.info("Failsafe reset")
            else:
                log.warning("Cannot reset failsafe — battery still critical")

    def get_status(self) -> dict:
        """Return all safety statuses as a dict."""
        with self._lock:
            return {
                "gps_ok": self._gps_ok,
                "battery_ok": self._battery_ok,
                "battery_V": round(self._battery_voltage, 1),
                "motors_ok": self._motors_ok,
                "failsafe": self._failsafe_triggered,
                "safe_to_fly": self._battery_ok and self._motors_ok,
            }
