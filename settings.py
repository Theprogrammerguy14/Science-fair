"""
DRONE AI — Global Settings
All tunable parameters in one place.
Raspberry Pi ONLY — no ESP32, no Arduino.
"""

# ──────────────────────────────────────────────
# AI / Detection
# ──────────────────────────────────────────────
MODEL_PATH = "models/detect.tflite"
LABELS_PATH = "models/labels.txt"
DETECTION_THRESHOLD = 0.55
HUMAN_CLASS_ID = 0                  # 'person' is class 0 in COCO
INPUT_SIZE = (300, 300)
MAX_DETECTIONS = 10

# ──────────────────────────────────────────────
# Camera
# ──────────────────────────────────────────────
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# ──────────────────────────────────────────────
# GPS (connected directly to Pi UART)
# ──────────────────────────────────────────────
GPS_PORT = "/dev/ttyAMA0"
GPS_BAUD = 9600

# ──────────────────────────────────────────────
# Motor PWM — 4 brushless ESCs via Pi GPIO
# Uses pigpio for hardware-timed PWM
# ──────────────────────────────────────────────
MOTOR_PINS = {
    "front_left":  12,   # BCM GPIO 12 (PWM0)
    "front_right": 13,   # BCM GPIO 13 (PWM1)
    "rear_left":   18,   # BCM GPIO 18 (PWM0)
    "rear_right":  19,   # BCM GPIO 19 (PWM1)
}
ESC_MIN_US = 1000       # Min pulse width (µs) — motors off
ESC_MAX_US = 2000       # Max pulse — full throttle
ESC_ARM_US = 1000       # Arming pulse
ESC_IDLE_US = 1100      # Idle spin after arming

# ──────────────────────────────────────────────
# Status LEDs & Buzzer & Kill Switch (BCM)
# ──────────────────────────────────────────────
LED_STATUS = 17          # Green — system OK
LED_DETECTION = 27       # Red   — human detected
LED_WARNING = 22         # Yellow — low battery
BUZZER_PIN = 23          # Buzzer for audible alert
KILL_SWITCH_PIN = 24     # Physical emergency button (pull-up, active LOW)

# ──────────────────────────────────────────────
# Battery Monitoring (MCP3008 ADC over SPI)
# ──────────────────────────────────────────────
BATTERY_ADC_CHANNEL = 0
BATTERY_VOLTAGE_DIVIDER = 4.0  # Voltage divider ratio
BATTERY_FULL_VOLTAGE = 12.6
BATTERY_WARN_VOLTAGE = 11.1
BATTERY_CRITICAL_VOLTAGE = 10.5
RETURN_HOME_RESERVE_PCT = 20

# ──────────────────────────────────────────────
# Mission Planner
# ──────────────────────────────────────────────
SEARCH_GRID_ROWS = 4
SEARCH_GRID_COLS = 4
GRID_CELL_SIZE_M = 10.0
HOVER_TIME_ON_DETECT_S = 5.0
DEFAULT_ALTITUDE_M = 10.0

# ──────────────────────────────────────────────
# Safety
# ──────────────────────────────────────────────
FAILSAFE_ACTION = "LAND"

# ──────────────────────────────────────────────
# Ground Station (Pi built-in WiFi)
# ──────────────────────────────────────────────
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 5000

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_DIR = "logs"
LOG_LEVEL = "INFO"
