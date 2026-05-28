# =============================================================================
#  PTZ Basketball Tracker  —  Configuration
#  Only edit this file.
# =============================================================================
from pathlib import Path

# Camera credentials & connection
# Dahua PTZ camera — IP 192.168.0.102
# RTSP sub-stream: channel=1&subtype=1
# ONVIF port on Dahua is 80
RTSP_URL = "rtsp://admin:123456@192.168.0.101:554/cam/realmonitor?channel=1&subtype=1"
CAM_IP   = "192.168.0.101"
CAM_PORT = 80
CAM_USER = "admin"
CAM_PASS = "123456"

# Detection — NanoDet-Plus ONNX  (416x416 = better accuracy than 320, still fast on CPU)
MODEL_PATH  = Path(r"C:\Tracking cam\ptz_group_tracker\models\nanodet-plus-m_416.onnx")
MODEL_URL   = "https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_416.onnx"
PERSON_CONF = 0.58  # higher = fewer false positives (chairs, signs, etc.)
NMS_IOU     = 0.55

# Person box sanity filters — reject detections that are clearly not a person.
# Measured in pixels on the *original* (pre-letterbox) frame.
#  PERSON_MIN_HEIGHT  — boxes shorter than this are noise (e.g. far-away objects
#                       mistaken for persons). Set ~3% of frame height at 1080p = 32 px.
#  PERSON_MAX_ASPECT  — width/height ratio cap. A standing person is never wider
#                       than tall; 0.85 rejects horizontal blobs (benches, ads).
PERSON_MIN_HEIGHT = 32    # pixels
PERSON_MAX_ASPECT = 0.85  # w/h; reject if box is wider than this ratio

# Low-light / noise-reduction — frame pre-processing
#  CLAHE_ENABLED  — apply CLAHE contrast enhancement on the L channel (LAB)
#                   before inference.  Helps the model in dark/evening frames.
#  CLAHE_CLIP     — contrast clip limit.  2.0 = moderate; 4.0 = aggressive.
#  CLAHE_GRID     — tile grid size in pixels.  8 = fine, 16 = coarser.
CLAHE_ENABLED = True
CLAHE_CLIP    = 2.0
CLAHE_GRID    = 8

# Temporal consistency — filter out single-frame noise blobs by requiring a
# detection to appear in several consecutive frames before the tracker sees it.
#
#  DETECTION_MIN_HITS — consecutive frames a box must appear before it is
#    forwarded to the tracker.
#    1 = disabled (pass-through, current behaviour)
#    2 = gentle — eliminates most flicker, ~1 frame extra lag on new players
#    3 = strong  — very clean, noticeable lag when a player enters the frame
#
#  DETECTION_MAX_AGE  — frames a confirmed detection is kept alive without a
#    new match (brief occlusion tolerance) before it is dropped.
DETECTION_MIN_HITS = 3
DETECTION_MAX_AGE  = 4


# ===================== TRACKING SETTINGS (ORDERED) =====================
# --- Smoothing and responsiveness ---
# EMA filter on the pan/tilt error.  Higher alpha = faster reaction, more jitter.
# The actual alpha used each frame is:
#       alpha = min(EMA_ALPHA_MAX, EMA_ALPHA + speed_norm * EMA_ALPHA_SCALE)
# so EMA_ALPHA is the *base* (slow-action) value and EMA_ALPHA_MAX is the
# ceiling allowed during fast action.  Keep EMA_ALPHA <= EMA_ALPHA_MAX.
EMA_ALPHA        = 0.16   # Base smoothing (must be <= EMA_ALPHA_MAX)
EMA_ALPHA_MAX    = 0.24   # Ceiling on fast action
EMA_ALPHA_SCALE  = 0.08   # Speed-to-alpha multiplier (adaptive boost)
VEL_EMA_ALPHA    = 0.10   # Smoothing for velocity estimate

# --- Velocity profile (how fast camera moves) ---
MAX_VEL    = 0.5    # Moderate cap for stability (raise if too slow)
PT_MIN_VEL = 0.05   # Slowest command sent
SLOW_ZONE  = 1.0    # Start slowing as soon as off-center (very smooth, no overshoot)
DEADBAND   = 0.38   # Wide stop zone for fast PTZ (stops easily)
START_BAND = 0.2    # Restart only when error exceeds 20% (quicker re-centering)

# --- Prediction and lead ---
LEAD_TIME  = 0.0    # Seconds to predict ahead (0 = no lead)

# --- Action weighting ---
ACTION_SIGMA        = 0.30   # How wide the "action zone" is (fraction of frame width)
MIN_ACTION_WEIGHT   = 0.15   # Fraction of max weight to count as "in action"

# --- Group clustering ---
# Two persons are considered the same group if their centres are closer
# than GROUP_CLUSTER_DIST (as a fraction of frame width).
#  0.25 = strict (must be quite close to be a group)
#  0.40 = loose (people farther apart still count as a group)
GROUP_CLUSTER_DIST  = 0.30

# How strongly the camera prefers larger groups over smaller ones.
# Each cluster pulls the camera with weight  (cluster_size ** GROUP_POWER).
# The camera target is a smooth weighted blend of all cluster centroids,
# so it never "jumps" when membership changes by one person.
#   1.0 = mild preference (3 vs 1 → 75% / 25%)
#   2.0 = clear preference (3 vs 1 → 90% / 10%)
#   3.0 = strong preference (3 vs 1 → 96% / 4%)   ← default
#   5.0 = near hard-switch  (3 vs 1 → 99.6% / 0.4%)
GROUP_POWER         = 3.0

# --- Pan/tilt coordination ---
PAN_PRIORITY_SCALE  = 1.2    # How strongly fast pan suppresses tilt EMA alpha

# --- Tilt-up bias (when person is very close) ---
TILT_BIAS_THRESHOLD = 0.9    # Person box height fraction to trigger tilt bias
TILT_UP_BIAS        = 1.0    # Max upward shift when overflowing
CLOSE_MAX_VEL       = 0.18   # Tilt speed cap when close

# --- Misc ---
COAST_SEC  = 3    # Keep moving N seconds after target disappears off screen (searching)

# Application
ENABLE_PTZ  = True
SHOW_WINDOW = True