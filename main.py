#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║  MODULAR AI-POWERED AUTONOMOUS DRONE FOR DISASTER RESPONSE     ║
║  Raspberry Pi — SINGLE BOARD Controller (no ESP32/Arduino)      ║
║                                                                  ║
║  Everything runs on the Pi:                                      ║
║    • AI inference (TFLite)                                       ║
║    • Motor control (pigpio PWM)                                  ║
║    • Battery monitoring (MCP3008 ADC)                            ║
║    • WiFi dashboard (Flask)                                      ║
║    • GPS (UART)                                                  ║
║    • LEDs + Buzzer (GPIO)                                        ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
  python3 main.py                 # Normal start (manual mode)
  python3 main.py --auto          # Auto-start autonomous mission
  python3 main.py --demo          # Demo mode (AI + dashboard, no motors)
"""

import sys
import os
import time
import signal
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings
from utils.logger import get_logger
from utils.gpio_helper import GPIOManager
from utils.gps_parser import GPSReader
from utils.battery_reader import BatteryReader
from modules.camera import Camera
from modules.ai_detector import AIDetector
from modules.mission_planner import MissionPlanner, FlightMode
from modules.energy_planner import EnergyPlanner
from modules.decision_engine import DecisionEngine
from modules.flight_controller import FlightController
from modules.safety_monitor import SafetyMonitor
from modules.heatmap_logger import HeatmapLogger
from modules.ground_station import GroundStation

log = get_logger("main")

running = True


def signal_handler(sig, frame):
    global running
    log.info("Shutdown signal received")
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ═══════════════════════════════════════════════════
#  INITIALIZATION
# ═══════════════════════════════════════════════════
def init_all(demo_mode: bool = False):
    log.info("=" * 60)
    log.info("  DRONE AI — System Initialization (Pi-Only)")
    log.info("=" * 60)

    modules = {}

    # 1. GPIO Manager
    log.info("[1/8] Setting up GPIO...")
    gpio = GPIOManager()
    gpio.setup()
    gpio.led_status(True)  # green LED on = booting
    modules["gpio"] = gpio

    # 2. Camera
    log.info("[2/8] Starting camera...")
    camera = Camera()
    if not camera.start():
        if not demo_mode:
            log.error("Camera failed — cannot continue")
            sys.exit(1)
        log.warning("Camera not available — demo mode uses test frames")
    modules["camera"] = camera

    # 3. AI Detector
    log.info("[3/8] Loading AI model...")
    detector = AIDetector()
    if not detector.load_model():
        log.error("AI model failed to load — check models/ directory")
        sys.exit(1)
    modules["detector"] = detector

    # 4. GPS Reader
    log.info("[4/8] Starting GPS...")
    gps = GPSReader()
    gps.start()
    modules["gps"] = gps

    # 5. Battery Reader
    log.info("[5/8] Starting battery monitor...")
    battery_reader = BatteryReader()
    battery_reader.start()
    modules["battery_reader"] = battery_reader

    # 6. Mission + Energy Planner
    log.info("[6/8] Initializing planners...")
    mission = MissionPlanner()
    energy = EnergyPlanner()
    modules["mission"] = mission
    modules["energy"] = energy

    # 7. Flight Controller + Safety
    log.info("[7/8] Initializing flight controller & safety...")
    safety = SafetyMonitor()
    decision = DecisionEngine(mission, energy)
    flight = FlightController(gpio)
    if not demo_mode:
        flight.start()
    modules["safety"] = safety
    modules["decision"] = decision
    modules["flight"] = flight

    # 8. Ground Station Dashboard
    log.info("[8/8] Starting ground station dashboard...")
    dashboard = GroundStation()
    dashboard.start()
    heatmap = HeatmapLogger()
    modules["dashboard"] = dashboard
    modules["heatmap"] = heatmap

    gpio.blink_led(settings.LED_STATUS, times=3, interval=0.15)

    log.info("=" * 60)
    log.info("  ALL SYSTEMS READY")
    log.info(f"  Dashboard: http://0.0.0.0:{settings.DASHBOARD_PORT}")
    log.info(f"  Mode: {'DEMO (no motors)' if demo_mode else 'LIVE'}")
    log.info("=" * 60)

    return modules


# ═══════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════
def main_loop(modules: dict, auto_start: bool = False, demo_mode: bool = False):
    global running

    camera: Camera = modules["camera"]
    detector: AIDetector = modules["detector"]
    gps: GPSReader = modules["gps"]
    battery_reader: BatteryReader = modules["battery_reader"]
    mission: MissionPlanner = modules["mission"]
    energy: EnergyPlanner = modules["energy"]
    decision: DecisionEngine = modules["decision"]
    flight: FlightController = modules["flight"]
    safety: SafetyMonitor = modules["safety"]
    dashboard: GroundStation = modules["dashboard"]
    heatmap: HeatmapLogger = modules["heatmap"]
    gpio: GPIOManager = modules["gpio"]

    prev_frame = None
    loop_count = 0

    # Auto-start mission
    if auto_start:
        fix = gps.fix
        if fix.has_fix:
            log.info("Auto-starting autonomous mission...")
            mission.start_mission(fix.latitude, fix.longitude)
            heatmap.start_recording()
        else:
            log.warning("No GPS fix — waiting...")

    log.info("Entering main loop (Ctrl+C to stop)...")

    while running:
        t_start = time.time()

        # ── Step 1: Capture ──────────────────────
        frame = camera.get_frame()
        frame_for_model = camera.get_frame_for_model()

        if frame_for_model is None:
            time.sleep(0.1)
            continue

        # ── Step 2: AI Inference ─────────────────
        detections = detector.detect(
            frame_for_model,
            frame_w=settings.CAMERA_WIDTH,
            frame_h=settings.CAMERA_HEIGHT,
        )

        # ── Step 3: Read Sensors ─────────────────
        fix = gps.fix
        safety.update_gps(fix.has_fix, fix.satellites)

        bat_v = battery_reader.voltage
        energy.update_voltage(bat_v)
        safety.update_battery(bat_v)

        # ── Step 4: Process Dashboard Commands ───
        cmd = dashboard.get_command()
        if cmd:
            _handle_command(cmd, mission, flight, heatmap, gps)

        # ── Step 5: Decision ─────────────────────
        result = decision.evaluate(
            detections=detections,
            gps_lat=fix.latitude,
            gps_lon=fix.longitude,
            frame=frame,
            prev_frame=prev_frame,
        )

        # ── Step 6: Act ──────────────────────────
        if not demo_mode:
            wp = mission.get_current_waypoint()
            flight.execute(result.flight_command, wp)

            if result.should_signal:
                flight.activate_signal(True)
            else:
                # Turn off detection LED after hover period
                gpio.led_detection(False)

        # Battery warning LED
        if energy.should_return_home():
            gpio.led_warning(True)
        else:
            gpio.led_warning(False)

        # ── Step 7: Log & Dashboard ──────────────
        # Heatmap
        if heatmap._recording:
            heatmap.log_position(
                fix.latitude, fix.longitude, fix.altitude,
                mission.state.mode.name,
            )
            heatmap.set_area_covered(mission.state.area_covered_pct)
            for d in detections:
                if d.is_human:
                    exp_text = result.explanations[0].summary if result.explanations else ""
                    heatmap.log_detection(
                        fix.latitude, fix.longitude,
                        d.class_name, d.confidence, exp_text,
                    )

        # Annotated video frame
        annotated = detector.draw_boxes(frame, detections) if frame is not None else None
        if annotated is not None:
            import cv2
            _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            dashboard.update_video(jpeg.tobytes())

        # Dashboard state
        dashboard.update("mode", mission.state.mode.name)
        dashboard.update("battery", energy.get_status_string())
        dashboard.update("gps",
            f"{fix.latitude:.6f}, {fix.longitude:.6f} "
            f"(sats: {fix.satellites}, alt: {fix.altitude:.1f}m)"
        )
        dashboard.update("mission_progress", round(mission.state.area_covered_pct))
        dashboard.update("alert", result.alert_message)
        dashboard.update("safety", safety.get_status())
        dashboard.update("inference_ms", round(detector.inference_time_ms))
        dashboard.update("motor_state", flight.state.name)
        dashboard.update("total_detections", decision.total_detections)

        if result.explanations:
            dashboard.update("explanations", [e.summary for e in result.explanations])
        else:
            dashboard.update("explanations", [])

        # ── Housekeeping ─────────────────────────
        prev_frame = frame
        loop_count += 1

        # Enforce ~15 Hz
        elapsed = time.time() - t_start
        sleep_time = max(0, (1.0 / 15) - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

        if loop_count % 100 == 0:
            fps = 1.0 / max(elapsed, 0.001)
            log.debug(
                f"Loop #{loop_count}: {elapsed*1000:.0f}ms "
                f"({fps:.1f} Hz) | AI: {detector.inference_time_ms:.0f}ms | "
                f"Detections: {len(detections)} | Bat: {bat_v:.1f}V"
            )


# ═══════════════════════════════════════════════════
#  COMMAND HANDLER
# ═══════════════════════════════════════════════════
def _handle_command(cmd: str, mission, flight, heatmap, gps):
    cmd = cmd.upper().strip()
    log.info(f"Executing command: {cmd}")

    if cmd == "ARM":
        flight.arm()
    elif cmd == "DISARM":
        flight.disarm()
    elif cmd == "TAKEOFF":
        flight.takeoff()
    elif cmd == "LAND":
        flight.land()
    elif cmd == "RTH":
        mission.trigger_return_home()
        flight.return_home()
    elif cmd == "EMERGENCY":
        flight.emergency_stop()
    elif cmd == "START_MISSION":
        fix = gps.fix
        mission.start_mission(fix.latitude, fix.longitude)
        heatmap.start_recording()
    elif cmd == "STOP_MISSION":
        heatmap.stop_recording()
        heatmap.save_json()
        heatmap.generate_heatmap()
    else:
        log.warning(f"Unknown command: {cmd}")


# ═══════════════════════════════════════════════════
#  SHUTDOWN
# ═══════════════════════════════════════════════════
def shutdown(modules: dict):
    log.info("Shutting down all systems...")

    heatmap: HeatmapLogger = modules.get("heatmap")
    if heatmap and heatmap._recording:
        heatmap.stop_recording()
        heatmap.save_json()
        heatmap.generate_heatmap()

    flight: FlightController = modules.get("flight")
    if flight:
        flight.stop()

    for name in ("battery_reader", "gps", "camera"):
        mod = modules.get(name)
        if mod and hasattr(mod, "stop"):
            mod.stop()

    gpio: GPIOManager = modules.get("gpio")
    if gpio:
        gpio.cleanup()

    log.info("All systems shut down. Goodbye.")


# ═══════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone AI — Raspberry Pi Controller")
    parser.add_argument("--auto", action="store_true", help="Auto-start autonomous mission")
    parser.add_argument("--demo", action="store_true", help="Demo mode (AI + dashboard, no motors)")
    args = parser.parse_args()

    if args.demo:
        log.info("*** DEMO MODE — motors disabled, AI + dashboard only ***")

    modules = init_all(demo_mode=args.demo)

    try:
        main_loop(modules, auto_start=args.auto, demo_mode=args.demo)
    except Exception as e:
        log.critical(f"Unhandled exception: {e}", exc_info=True)
    finally:
        shutdown(modules)
