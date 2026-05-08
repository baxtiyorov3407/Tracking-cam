# =============================================================================
#  PTZ Basketball Tracker  —  Configuration
#  Only edit this file.
# =============================================================================
from pathlib import Path

# Camera credentials & connectionq
# Dahua PTZ camera — IP 192.168.219.33
# RTSP sub-stream: channel=1&subtype=1
# ONVIF port on Dahua is 80
RTSP_URL = "rtsp://admin:123456@192.168.219.33:554/cam/realmonitor?channel=1&subtype=1"
CAM_IP   = "192.168.219.33"
CAM_PORT = 80
CAM_USER = "admin"
CAM_PASS = "123456"

# Detection — NanoDet-Plus ONNX (5-10x faster than YOLO nano on CPU)
MODEL_PATH  = Path(r"C:\Tracking cam\ptz_group_tracker\models\nanodet-plus-m_320.onnx")
MODEL_URL   = "https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_320.onnx"
PERSON_CONF = 0.35
BALL_CONF   = 0.22
NMS_IOU     = 0.55

# Tracking
EMA_ALPHA   = 0.20   # lower = more smoothing, fewer jitter-driven direction flips
BALL_WEIGHT = 0.65   # 0.0 = group centre only, 1.0 = ball centre only

# Velocity profile  (all values = fraction of half-frame width)
#
#  |error|  >=  SLOW_ZONE  →  MAX_VEL
#  DEADBAND  <  |error|  <  SLOW_ZONE  →  linear ramp PT_MIN_VEL..MAX_VEL
#  |error|  <=  DEADBAND  →  stop
#
#  MAX_VEL caps how fast the motor ever runs — key overshoot lever.
#  With ~1s latency keep MAX_VEL low (0.25–0.40) until tracking feels stable.
DEADBAND   = 0.12   # stop within 12% of frame centre
START_BAND = 0.22   # restart only when error exceeds 22%
SLOW_ZONE  = 1.0   # ramp zone: 12% → 100%; full (capped) speed beyond 100%
MAX_VEL    = 1.0   # hard cap on motor speed  (1.0 = max, 0.35 = gentle)
PT_MIN_VEL = 0.12   # slowest command sent (prevents motor stall hum)

COAST_SEC  = 5.0    # keep moving N seconds after target disappears off screen

# Application
ENABLE_PTZ  = True
SHOW_WINDOW = True