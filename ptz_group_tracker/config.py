# ──────────────────────────────────────────────────────────────────────────────
#  PTZ Group Tracker – Configuration
#  These are ALL the settings you need to tune.
# ──────────────────────────────────────────────────────────────────────────────

# ── Camera ────────────────────────────────────────────────────────────────────
RTSP_URL = "rtsp://admin:123456@192.168.219.33:554/stream2"
CAM_IP   = "192.168.219.33"
CAM_PORT = 80
CAM_USER = "admin"
CAM_PASS = "123456"

# ── Detection ─────────────────────────────────────────────────────────────────
YOLO_MODEL  = "yolo26n.pt"   # nano = fastest on CPU; try yolo26m.pt for accuracy
CONF_THRESH = 0.40           # detection confidence threshold (0.0–1.0)

# ── Group framing ─────────────────────────────────────────────────────────────
TARGET_FILL   = 0.72   # target: group should fill 72% of the larger frame dimension
GROUP_PADDING = 0.10   # add 10% padding around the detected group bounding box

# ── Pan / Tilt controller ─────────────────────────────────────────────────────
KP_PAN  = 0.70   # pan  proportional gain
KP_TILT = 0.70   # tilt proportional gain

EMA_ALPHA = 0.90   # error smoothing (higher = more reactive to fast movement)

DEADBAND   = 0.1   # stop motor when within 6% of centre (tighter = more accurate)
START_BAND = 0.10   # restart motor when error grows above 10%

PT_MIN_VEL = 0.08   # minimum motor speed (below this = Stop command)

# How long (seconds) the camera keeps moving after losing the target.
# Helps when a player runs off the edge — camera chases them instead of stopping.
COAST_DURATION = 1.5

# ── Flags ─────────────────────────────────────────────────────────────────────
ENABLE_PTZ        = True
ENABLE_AUDIO      = False   # True = stream audio via ffplay (requires FFmpeg in PATH)
SHOW_DEBUG_WINDOW = True