#!/bin/bash
# ============================================================
# DRONE AI — Raspberry Pi Setup (Debian Trixie / Bookworm)
# Run: chmod +x install.sh && ./install.sh
# ============================================================
set -e
echo "========================================"
echo "  DRONE AI — Raspberry Pi Installer"
echo "  (Single Board — Pi handles everything)"
echo "========================================"

# Detect Debian version
DEBIAN_VER=$(cat /etc/debian_version 2>/dev/null || echo "unknown")
echo "  Detected Debian: $DEBIAN_VER"

# ──────────────────────────────────────────────
# 1. System packages (Trixie-compatible)
# ──────────────────────────────────────────────
echo "[1/6] Installing system packages..."
sudo apt-get update
sudo apt-get install -y \
    python3-pip python3-venv python3-dev \
    libopenblas-dev \
    libopenjp2-7 \
    libtiff-dev \
    python3-opencv \
    python3-lgpio \
    python3-gpiozero \
    python3-spidev \
    wget unzip

# ──────────────────────────────────────────────
# 2. Python virtual environment
# ──────────────────────────────────────────────
echo "[2/6] Creating Python virtual environment..."
python3 -m venv ~/drone_env --system-site-packages
source ~/drone_env/bin/activate

# ──────────────────────────────────────────────
# 3. Python packages
# ──────────────────────────────────────────────
echo "[3/6] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# tflite-runtime — try new name first, then old, then full tensorflow
echo "  Installing TFLite runtime..."
pip install ai-edge-litert 2>/dev/null \
    && echo "  → Installed ai-edge-litert (recommended)" \
    || pip install tflite-runtime 2>/dev/null \
    && echo "  → Installed tflite-runtime" \
    || pip install tensorflow 2>/dev/null \
    && echo "  → Installed full tensorflow (fallback)" \
    || echo "  ⚠ TFLite not installed. Run: pip install ai-edge-litert"

# ──────────────────────────────────────────────
# 4. Enable Pi interfaces
# ──────────────────────────────────────────────
echo "[4/6] Enabling camera, SPI, UART..."
sudo raspi-config nonint do_camera 0   2>/dev/null || true
sudo raspi-config nonint do_spi 0      2>/dev/null || true
sudo raspi-config nonint do_serial_hw 0 2>/dev/null || true
sudo raspi-config nonint do_i2c 0      2>/dev/null || true

# ──────────────────────────────────────────────
# 5. Download AI model
# ──────────────────────────────────────────────
echo "[5/6] Downloading AI model..."
MODEL_DIR="models"
mkdir -p $MODEL_DIR

if [ ! -f "$MODEL_DIR/detect.tflite" ]; then
    echo "  Downloading MobileNet-SSD..."
    TMP_ZIP="/tmp/coco_ssd.zip"
    wget -q -O "$TMP_ZIP" \
        "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip" \
        && unzip -o "$TMP_ZIP" -d "$MODEL_DIR" \
        && rm -f "$TMP_ZIP" \
        && echo "  → Model downloaded and extracted" \
        || echo "  → Download failed. Place your .tflite model in models/"
fi

# ──────────────────────────────────────────────
# 6. Labels file
# ──────────────────────────────────────────────
echo "[6/6] Creating labels file..."
cat > "$MODEL_DIR/labels.txt" << 'EOF'
person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
backpack
umbrella
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
couch
potted plant
bed
dining table
toilet
tv
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
book
clock
vase
scissors
teddy bear
hair drier
toothbrush
EOF

echo ""
echo "========================================"
echo "  Setup complete!"
echo ""
echo "  Activate:  source ~/drone_env/bin/activate"
echo "  Run:       python3 main.py --demo"
echo "  Dashboard: http://\$(hostname -I | awk '{print \$1}'):5000"
echo "========================================"
