"""
DRONE AI — Flight Controller (Direct Pi GPIO)
Controls 4 brushless ESCs directly from the Raspberry Pi using pigpio
for hardware-timed PWM signals.

No Arduino needed — the Pi handles everything.

Safety: Kill switch on GPIO triggers immediate motor shutoff.

IMPORTANT: This is a simplified controller for the science fair.
A production flight controller would use a PID loop with IMU feedback.
For the competition, you can also plug in a Pixhawk/Betaflight FC
via serial and send MAVLink commands instead — see the comments below.
"""

import time
import threading
from enum import Enum, auto
from utils.gpio_helper import GPIOManager
from utils.logger import get_logger
from config import settings

log = get_logger("flight_ctrl")


class MotorState(Enum):
    DISARMED = auto()
    ARMED = auto()
    FLYING = auto()
    LANDING = auto()


class FlightController:
    """
    Direct motor control via Pi GPIO.
    
    Motor layout (top view, front = up):
        FL ↺   ↻ FR       (FL, RR spin CCW)
           \\ /           (FR, RL spin CW)
            X
           / \\
        RL ↻   ↺ RR
    """

    def __init__(self, gpio: GPIOManager):
        self._gpio = gpio
        self._state = MotorState.DISARMED
        self._throttle = {name: settings.ESC_MIN_US for name in settings.MOTOR_PINS}
        self._base_throttle = settings.ESC_MIN_US
        self._lock = threading.Lock()

    # ── Lifecycle ────────────────────────────────
    def start(self):
        """Initialize ESCs (send min throttle for calibration)."""
        log.info("Initializing ESCs...")
        for name, pin in settings.MOTOR_PINS.items():
            self._gpio.set_motor_us(pin, settings.ESC_MIN_US)
        time.sleep(2)  # ESCs need ~2s at min throttle to initialize
        log.info("ESCs initialized — ready to arm")

        # Register kill switch
        self._gpio.register_kill_switch(self.emergency_stop)

    def stop(self):
        """Disarm and release motors."""
        self.disarm()

    # ── Basic Commands ───────────────────────────
    def arm(self) -> bool:
        """Arm the motors (idle spin)."""
        if self._state != MotorState.DISARMED:
            log.warning(f"Cannot arm — current state: {self._state.name}")
            return False

        log.info("ARMING motors...")
        for name, pin in settings.MOTOR_PINS.items():
            self._gpio.set_motor_us(pin, settings.ESC_ARM_US)
        time.sleep(1)

        # Bring to idle
        for name, pin in settings.MOTOR_PINS.items():
            self._gpio.set_motor_us(pin, settings.ESC_IDLE_US)
            self._throttle[name] = settings.ESC_IDLE_US

        self._state = MotorState.ARMED
        self._gpio.led_status(True)
        log.info("Motors ARMED")
        return True

    def disarm(self) -> bool:
        """Disarm motors — set all to minimum."""
        log.info("DISARMING motors...")
        for name, pin in settings.MOTOR_PINS.items():
            self._gpio.set_motor_us(pin, settings.ESC_MIN_US)
            self._throttle[name] = settings.ESC_MIN_US

        self._state = MotorState.DISARMED
        self._gpio.led_status(False)
        log.info("Motors DISARMED")
        return True

    def takeoff(self, target_alt_m: float = None) -> bool:
        """
        Gradual throttle increase for takeoff.
        In a real system, this would use a barometer/IMU PID loop.
        """
        if self._state == MotorState.DISARMED:
            self.arm()

        target = target_alt_m or settings.DEFAULT_ALTITUDE_M
        log.info(f"Taking off to {target}m...")
        self._state = MotorState.FLYING

        # Ramp throttle up gradually (simplified — real FC uses PID)
        takeoff_throttle = int(settings.ESC_MIN_US + (settings.ESC_MAX_US - settings.ESC_MIN_US) * 0.55)
        self._set_all_motors(takeoff_throttle)
        self._gpio.led_status(True)
        return True

    def land(self) -> bool:
        """Gradual descent and disarm."""
        log.info("LANDING...")
        self._state = MotorState.LANDING

        # Gradually reduce throttle
        current = self._base_throttle
        while current > settings.ESC_IDLE_US:
            current = max(settings.ESC_IDLE_US, current - 20)
            self._set_all_motors(current)
            time.sleep(0.1)

        time.sleep(1)
        self.disarm()
        log.info("Landed and disarmed")
        return True

    def hover(self) -> bool:
        """Maintain current altitude (simplified — constant throttle)."""
        if self._state == MotorState.FLYING:
            hover_throttle = int(settings.ESC_MIN_US + (settings.ESC_MAX_US - settings.ESC_MIN_US) * 0.50)
            self._set_all_motors(hover_throttle)
        return True

    def set_throttle(self, percentage: float):
        """Set all motors to a percentage (0–100) of max throttle."""
        pct = max(0, min(100, percentage))
        us = int(settings.ESC_MIN_US + (settings.ESC_MAX_US - settings.ESC_MIN_US) * (pct / 100))
        self._set_all_motors(us)

    def emergency_stop(self):
        """IMMEDIATE motor cutoff — no gradual descent."""
        log.critical("EMERGENCY STOP — all motors OFF")
        self._gpio.stop_all_motors()
        self._state = MotorState.DISARMED
        self._gpio.buzzer_beep(duration=1.0, times=5)
        self._gpio.blink_led(settings.LED_WARNING, times=10, interval=0.1)

    # ── Navigation Commands (simplified) ─────────
    def fly_to(self, lat: float, lon: float, alt: float = None):
        """
        Navigate to a GPS waypoint.
        
        NOTE: Real GPS-guided flight requires:
          1. GPS + compass (magnetometer) for heading
          2. IMU (accelerometer + gyroscope) for attitude
          3. PID controller to adjust individual motor speeds
          
        For the science fair, you can:
          a) Demo the AI + detection part (--demo mode) without flying
          b) Use a Pixhawk flight controller for actual flight and
             send MAVLink waypoint commands via dronekit-python
        """
        if self._state != MotorState.FLYING:
            log.debug("Not flying — ignoring fly_to command")
            return

        alt = alt or settings.DEFAULT_ALTITUDE_M
        log.info(f"Navigating to ({lat:.6f}, {lon:.6f}) alt={alt:.0f}m")
        # In demo mode, we just maintain hover
        self.hover()

    def return_home(self):
        """Return to home position."""
        log.info("Returning to home...")
        self.hover()  # simplified — real implementation navigates back

    # ── Signal (buzzer + LED for detected humans) ─
    def activate_signal(self, on: bool = True):
        """Turn on/off the detection signal (buzzer + red LED)."""
        if on:
            self._gpio.led_detection(True)
            self._gpio.buzzer_beep(duration=0.3, times=5)
            log.info("Detection signal ACTIVE")
        else:
            self._gpio.led_detection(False)
            self._gpio.buzzer_off()

    # ── Execute Decision ─────────────────────────
    def execute(self, flight_command: str, waypoint=None):
        """
        Translate a DecisionEngine command into motor actions.
        Called by main loop after each decision cycle.
        """
        if flight_command == "HOLD":
            self.hover()
        elif flight_command == "HOVER":
            self.hover()
        elif flight_command == "FLY_TO_WP" and waypoint:
            self.fly_to(waypoint.latitude, waypoint.longitude, waypoint.altitude)
        elif flight_command == "LAND":
            self.land()
        elif flight_command == "RTH":
            self.return_home()
        elif flight_command == "EMERGENCY":
            self.emergency_stop()

    # ── Internal ─────────────────────────────────
    def _set_all_motors(self, pulse_us: int):
        """Set all four motors to the same pulse width."""
        with self._lock:
            self._base_throttle = pulse_us
            for name, pin in settings.MOTOR_PINS.items():
                self._gpio.set_motor_us(pin, pulse_us)
                self._throttle[name] = pulse_us

    # ── Properties ───────────────────────────────
    @property
    def is_armed(self) -> bool:
        return self._state != MotorState.DISARMED

    @property
    def state(self) -> MotorState:
        return self._state

    @property
    def throttle_values(self) -> dict:
        with self._lock:
            return dict(self._throttle)
