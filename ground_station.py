"""
DRONE AI — Ground Station Dashboard (Pi-Only)
Flask web dashboard served directly from the Pi's built-in WiFi.

Features:
  - Live camera feed with AI bounding boxes
  - Detection alerts + XAI explanations
  - Battery, GPS, mission status
  - Command buttons (ARM, TAKEOFF, LAND, RTH, EMERGENCY)
  - Heatmap view after mission

Access from any phone/laptop: http://<pi-ip>:5000
"""

import time
import threading
import json
from flask import Flask, Response, render_template_string, request, jsonify
from utils.logger import get_logger
from config import settings

log = get_logger("ground_station")

app = Flask(__name__)

# ── Shared State ─────────────────────────────────
_shared_state = {
    "frame_jpeg": None,
    "detections": [],
    "explanations": [],
    "battery": "",
    "gps": "",
    "mode": "INIT",
    "mission_progress": 0,
    "alert": "",
    "safety": {},
    "inference_ms": 0,
    "motor_state": "DISARMED",
    "total_detections": 0,
}
_state_lock = threading.Lock()
_command_queue: list[str] = []
_cmd_lock = threading.Lock()


def update_dashboard(key: str, value):
    with _state_lock:
        _shared_state[key] = value


def update_frame(jpeg_bytes: bytes):
    with _state_lock:
        _shared_state["frame_jpeg"] = jpeg_bytes


def get_pending_command() -> str | None:
    """Pop the next command from the queue (called by main loop)."""
    with _cmd_lock:
        if _command_queue:
            return _command_queue.pop(0)
    return None


# ── Video Stream ─────────────────────────────────
def _generate_frames():
    while True:
        with _state_lock:
            frame = _shared_state["frame_jpeg"]
        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        time.sleep(0.05)


@app.route("/video_feed")
def video_feed():
    return Response(
        _generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/status")
def api_status():
    with _state_lock:
        data = dict(_shared_state)
        del data["frame_jpeg"]  # don't send binary in JSON
    return jsonify(data)


@app.route("/api/command", methods=["POST"])
def api_command():
    """Receive commands from the dashboard buttons."""
    cmd = request.json.get("cmd", "").upper().strip()
    if cmd:
        with _cmd_lock:
            _command_queue.append(cmd)
        log.info(f"Dashboard command queued: {cmd}")
        return jsonify({"ok": True, "cmd": cmd})
    return jsonify({"ok": False}), 400


# ── Dashboard HTML ───────────────────────────────
DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DRONE AI — Ground Station</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:#0a0a0a;color:#e0e0e0;font-family:'Courier New',monospace}
  .header{background:#1a1a2e;padding:10px 16px;display:flex;justify-content:space-between;align-items:center;border-bottom:2px solid #00ff88}
  .header h1{color:#00ff88;font-size:1.1em}
  .badge{padding:3px 10px;border-radius:10px;font-size:.75em;font-weight:700}
  .badge-ok{background:#00ff88;color:#000}
  .badge-warn{background:#ffaa00;color:#000}
  .badge-crit{background:#ff3333;color:#fff}
  .main{display:grid;grid-template-columns:2fr 1fr;gap:10px;padding:10px;height:calc(100vh - 52px)}
  .video{background:#111;border:1px solid #333;border-radius:6px;overflow:hidden;position:relative}
  .video img{width:100%;height:100%;object-fit:contain}
  .overlay{position:absolute;top:6px;left:6px;background:rgba(0,0,0,.7);color:#0cf;padding:3px 8px;border-radius:4px;font-size:.75em}
  .side{display:flex;flex-direction:column;gap:8px;overflow-y:auto}
  .card{background:#1a1a1a;border:1px solid #333;border-radius:6px;padding:10px}
  .card h3{color:#00ff88;font-size:.8em;margin-bottom:6px;border-bottom:1px solid #222;padding-bottom:3px}
  .val{font-size:.95em;color:#fff}
  .alert-active{border-color:#ff3333;background:#1a0a0a}
  .alert-active .val{color:#ff6666}
  .bar{background:#333;border-radius:3px;height:16px;margin-top:4px}
  .bar-fill{background:linear-gradient(90deg,#00ff88,#0cf);height:100%;border-radius:3px;transition:width .5s}
  .explanation{background:#0d1a0d;border-left:3px solid #00ff88;padding:6px 8px;margin:3px 0;font-size:.78em}
  .safety-grid{display:grid;grid-template-columns:1fr 1fr;gap:3px}
  .si{padding:3px 6px;border-radius:3px;font-size:.72em;text-align:center}
  .safe{background:#0d2e0d;color:#00ff88}
  .unsafe{background:#2e0d0d;color:#ff3333}
  /* Command buttons */
  .cmd-panel{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}
  .cmd-btn{padding:8px 14px;border:none;border-radius:4px;font-family:inherit;font-size:.8em;font-weight:700;cursor:pointer;transition:opacity .15s}
  .cmd-btn:active{opacity:.7}
  .btn-arm{background:#00ff88;color:#000}
  .btn-takeoff{background:#0cf;color:#000}
  .btn-land{background:#ffaa00;color:#000}
  .btn-rth{background:#aa88ff;color:#000}
  .btn-emergency{background:#ff2222;color:#fff;font-size:.9em}
  .btn-mission{background:#4488ff;color:#fff}
  @media(max-width:700px){.main{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="header">
  <h1>&#x2B21; DRONE AI — Ground Station</h1>
  <span id="modeBadge" class="badge badge-ok">INIT</span>
</div>
<div class="main">
  <div class="video">
    <img src="/video_feed" alt="Live Feed">
    <div class="overlay" id="inferTime">-- ms</div>
  </div>
  <div class="side">
    <!-- COMMANDS -->
    <div class="card">
      <h3>&#x1F3AE; Commands</h3>
      <div class="cmd-panel">
        <button class="cmd-btn btn-arm" onclick="sendCmd('ARM')">ARM</button>
        <button class="cmd-btn btn-arm" onclick="sendCmd('DISARM')">DISARM</button>
        <button class="cmd-btn btn-takeoff" onclick="sendCmd('TAKEOFF')">TAKEOFF</button>
        <button class="cmd-btn btn-land" onclick="sendCmd('LAND')">LAND</button>
        <button class="cmd-btn btn-rth" onclick="sendCmd('RTH')">RTH</button>
        <button class="cmd-btn btn-mission" onclick="sendCmd('START_MISSION')">START MISSION</button>
        <button class="cmd-btn btn-mission" onclick="sendCmd('STOP_MISSION')">STOP MISSION</button>
        <button class="cmd-btn btn-emergency" onclick="if(confirm('EMERGENCY STOP?'))sendCmd('EMERGENCY')">&#x26A0; EMERGENCY STOP</button>
      </div>
    </div>
    <!-- ALERT -->
    <div class="card" id="alertCard">
      <h3>&#x26A0; Alert</h3>
      <div class="val" id="alertTxt">No alerts</div>
    </div>
    <!-- BATTERY -->
    <div class="card">
      <h3>&#x1F50B; Battery</h3>
      <div class="val" id="batTxt">--</div>
    </div>
    <!-- GPS -->
    <div class="card">
      <h3>&#x1F4CD; GPS</h3>
      <div class="val" id="gpsTxt">--</div>
    </div>
    <!-- MISSION -->
    <div class="card">
      <h3>&#x1F3AF; Mission</h3>
      <div class="bar"><div class="bar-fill" id="progBar" style="width:0%"></div></div>
      <div class="val" id="progTxt" style="margin-top:3px">0% — Detections: 0</div>
    </div>
    <!-- XAI -->
    <div class="card">
      <h3>&#x1F9E0; AI Explanations</h3>
      <div id="xaiDiv"><em>Waiting for detections...</em></div>
    </div>
    <!-- SAFETY -->
    <div class="card">
      <h3>&#x1F6E1; Safety</h3>
      <div class="safety-grid" id="safetyGrid"></div>
    </div>
  </div>
</div>
<script>
function sendCmd(cmd){
  fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cmd})})
  .then(r=>r.json()).then(d=>console.log('CMD:',d)).catch(e=>console.error(e));
}
function poll(){
  fetch('/api/status').then(r=>r.json()).then(d=>{
    document.getElementById('modeBadge').textContent=d.mode||'--';
    document.getElementById('batTxt').textContent=d.battery||'--';
    document.getElementById('gpsTxt').textContent=d.gps||'--';
    document.getElementById('progBar').style.width=(d.mission_progress||0)+'%';
    document.getElementById('progTxt').textContent=(d.mission_progress||0)+'% — Detections: '+(d.total_detections||0);
    document.getElementById('inferTime').textContent=(d.inference_ms||0)+' ms';
    const ac=document.getElementById('alertCard'),at=document.getElementById('alertTxt');
    at.textContent=d.alert||'No alerts';
    ac.className=d.alert?'card alert-active':'card';
    const xd=document.getElementById('xaiDiv');
    if(d.explanations&&d.explanations.length)
      xd.innerHTML=d.explanations.map(e=>'<div class="explanation">'+e+'</div>').join('');
    const sg=document.getElementById('safetyGrid');
    if(d.safety)
      sg.innerHTML=Object.entries(d.safety).map(([k,v])=>'<div class="si '+(v?'safe':'unsafe')+'">'+k.replace(/_/g,' ')+'</div>').join('');
  }).catch(()=>{});
  setTimeout(poll,500);
}
poll();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


# ── Server ───────────────────────────────────────
class GroundStation:
    def __init__(self):
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log.info(f"Dashboard: http://0.0.0.0:{settings.DASHBOARD_PORT}")

    def _run(self):
        app.run(
            host=settings.DASHBOARD_HOST,
            port=settings.DASHBOARD_PORT,
            debug=False, use_reloader=False, threaded=True,
        )

    def update(self, key: str, value):
        update_dashboard(key, value)

    def update_video(self, jpeg_bytes: bytes):
        update_frame(jpeg_bytes)

    def get_command(self) -> str | None:
        return get_pending_command()
