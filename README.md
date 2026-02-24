# MODULAR AI-POWERED AUTONOMOUS DRONE FOR DISASTER RESPONSE
## Raspberry Pi — Single Board Controller

**Everything runs on one Raspberry Pi. No ESP32. No Arduino.**

### Architecture
```
┌─────────────────────────────────────────┐
│              RASPBERRY PI               │
│                                         │
│  Camera ──► AI Detector (TFLite)        │
│                  │                      │
│          Decision Engine                │
│           ╱    │    ╲                   │
│      GPS    Energy    Safety            │
│     (UART)  Planner   Monitor           │
│               │                         │
│         Motor Control (pigpio PWM)      │
│          ┌──┴──┐                        │
│         ESC   ESC   ESC   ESC           │
│          │     │     │     │            │
│         M1    M2    M3    M4            │
│                                         │
│  Battery ◄── MCP3008 ADC (SPI)         │
│  LEDs/Buzzer ◄── GPIO                  │
│  WiFi Dashboard ◄── Flask (port 5000)  │
└─────────────────────────────────────────┘
```

### Directory Structure
```
drone_ai/
├── main.py                    # Entry point — boots everything
├── config/
│   └── settings.py            # All pins, thresholds, parameters
├── modules/
│   ├── camera.py              # Threaded camera capture
│   ├── ai_detector.py         # TFLite MobileNet-SSD inference
│   ├── explainable_ai.py      # XAI — explains WHY detections happen
│   ├── mission_planner.py     # Autonomous grid search
│   ├── energy_planner.py      # Battery-aware flight adaptation
│   ├── decision_engine.py     # Core IF/THEN mission logic
│   ├── flight_controller.py   # Direct motor PWM control via pigpio
│   ├── safety_monitor.py      # Battery/GPS/motor safety checks
│   ├── heatmap_logger.py      # Mission replay + detection heatmap
│   └── ground_station.py      # Flask web dashboard with controls
├── utils/
│   ├── gpio_helper.py         # LEDs, buzzer, kill switch, motor PWM
│   ├── gps_parser.py          # NMEA GPS parser
│   ├── battery_reader.py      # MCP3008 ADC battery voltage reader
│   └── logger.py              # Timestamped logging
├── models/                    # Place your .tflite model here
├── logs/                      # Auto-generated mission logs + heatmaps
├── requirements.txt
└── install.sh                 # One-command Pi setup
```

### Quick Start
```bash
chmod +x install.sh && ./install.sh
source ~/drone_env/bin/activate
python3 main.py --demo          # Demo: AI + dashboard, no motors
python3 main.py                 # Live: full system
python3 main.py --auto          # Auto-start autonomous grid search
```

### Dashboard
Open `http://<pi-ip>:5000` on your phone or laptop.
Includes: live AI video, command buttons, battery, GPS, XAI explanations, safety status.

### Hardware Wiring
| Component | Pi Pin (BCM) | Notes |
|---|---|---|
| ESC Front-Left | GPIO 12 | PWM0 — pigpio servo pulse |
| ESC Front-Right | GPIO 13 | PWM1 |
| ESC Rear-Left | GPIO 18 | PWM0 |
| ESC Rear-Right | GPIO 19 | PWM1 |
| Green LED | GPIO 17 | System OK |
| Red LED | GPIO 27 | Human detected |
| Yellow LED | GPIO 22 | Low battery |
| Buzzer | GPIO 23 | Alert sound |
| Kill Switch | GPIO 24 | Pull-up, active LOW |
| GPS TX | GPIO 15 (RXD) | /dev/ttyAMA0 @ 9600 |
| MCP3008 ADC | SPI0 (GPIO 8,9,10,11) | Battery voltage |
| Camera | CSI or USB | /dev/video0 |
