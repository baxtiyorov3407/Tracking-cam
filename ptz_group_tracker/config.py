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

# Detection — NanoDet-Plus ONNX  (416x416 = better accuracy than 320, still fast on CPU)
MODEL_PATH  = Path(r"C:\Tracking cam\ptz_group_tracker\models\nanodet-plus-m_416.onnx")
MODEL_URL   = "https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_416.onnx"
PERSON_CONF = 0.58   # higher = fewer false positives (chairs, signs, etc.)
BALL_CONF   = 0.22
NMS_IOU     = 0.55

# Person box sanity filters — reject detections that are clearly not a person.
# Measured in pixels on the *original* (pre-letterbox) frame.
#  PERSON_MIN_HEIGHT  — boxes shorter than this are noise (e.g. far-away objects
#                       mistaken for persons). Set ~3% of frame height at 1080p = 32 px.
#  PERSON_MAX_ASPECT  — width/height ratio cap. A standing person is never wider
#                       than tall; 0.85 rejects horizontal blobs (benches, ads).
PERSON_MIN_HEIGHT = 32    # pixels
PERSON_MAX_ASPECT = 0.85  # w/h; reject if box is wider than this ratio

# Tracking — core
EMA_ALPHA   = 0.14   # base smoothing (lower = smoother; higher = snappier)
BALL_WEIGHT = 0.5   # 0.0 = group centre only, 1.0 = ball centre only

# Pan-priority — when horizontal movement is fast, tilt response is throttled
# so the camera catches the person sideways first, then eases tilt in smoothly.
#  PAN_PRIORITY_SCALE — how strongly fast pan suppresses tilt EMA alpha.
#    0.0 = no suppression (old behaviour)
#    1.5 = strong: at pan speed 0.4 frame/s, tilt alpha drops to ~40% of normal
#    2.5 = very strong: tilt almost frozen during a fast break
PAN_PRIORITY_SCALE = 2.8

# Tilt-up bias — only activates when a person is so close they cannot fit
# vertically in the frame, so the camera tilts up to show the head.
#
#  TILT_BIAS_THRESHOLD — person box height as a fraction of frame height at
#    which the bias starts. Below this the person fits and stays centred.
#    0.85 = bias kicks in only when player fills >85% of frame height.
#  TILT_UP_BIAS — maximum upward shift in normalised tilt units when the
#    person fully overflows (box height >= frame height).
#  CLOSE_MAX_VEL — tilt motor speed cap in overflow zone. Prevents the
#    camera hunting/oscillating when a person fills the frame.
#    0.18 = very gentle tilt when close; raise if it feels too sluggish.
TILT_BIAS_THRESHOLD = 0.9
TILT_UP_BIAS        = 1.0
CLOSE_MAX_VEL       = 0.18

# NBA-style action-weighted tracking
#
#  ACTION_SIGMA   — how wide the "action zone" is, as a fraction of frame width.
#                   exp(-dist/sigma): players at sigma distance = 37% weight,
#                   at 2x sigma = 14%, at 3x sigma = 5%.
#                   0.30 works well for a side-wall cam covering half-court.
ACTION_SIGMA      = 0.30

#  MIN_ACTION_WEIGHT — fraction of the highest-weighted player a person must
#                   reach to be counted as "in the action" for the box overlay.
#                   0.15 = only players within roughly 2x sigma of the ball.
MIN_ACTION_WEIGHT = 0.15

# Lead prediction — camera aims ahead of the action centre
#  LEAD_TIME     — seconds to predict ahead (e.g. 0.35 s at fast-break speed
#                  ~40% frame/sec gives ~14% frame pre-lead).
LEAD_TIME         = 0.5

#  VEL_EMA_ALPHA — smoothing for the velocity estimate (higher = velocity
#                  estimate reacts faster to acceleration; lower = smoother).
VEL_EMA_ALPHA     = 0.38

# Adaptive EMA — EMA alpha scales with action speed so the camera reacts
# faster during fast-breaks and smoother during half-court sets.
#  EMA_ALPHA_MAX  — ceiling on adaptive alpha.
#  EMA_ALPHA_SCALE — speed-to-alpha multiplier (speed is in frame-widths/sec).
EMA_ALPHA_MAX     = 0.55
EMA_ALPHA_SCALE   = 3.20

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