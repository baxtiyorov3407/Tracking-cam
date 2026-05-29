# =============================================================================
#  PTZ Basketball Tracker  —  Configuration
#  Only edit this file.
# =============================================================================
from pathlib import Path

# Camera credentials & connection
# Dahua PTZ camera — IP 192.168.1.101
# RTSP sub-stream: channel=1&subtype=1
# ONVIF port on Dahua is 80
RTSP_URL = "rtsp://admin:123456@192.168.1.101:554/cam/realmonitor?channel=1&subtype=1"
CAM_IP   = "192.168.1.101"
CAM_PORT = 80
CAM_USER = "admin"
CAM_PASS = "123456"

# Detection — NanoDet-Plus ONNX (416x416, COCO mAP ~30.4). Reverted from the
# 1.5x variant to the baseline model that historically worked best.
MODEL_PATH  = Path(__file__).resolve().parent / "models" / "nanodet-plus-m_416.onnx"
MODEL_URL   = "https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_416.onnx"
PERSON_CONF = 0.58  # higher = fewer false positives (chairs, signs, etc.)
NMS_IOU     = 0.55

# Person box sanity filters — reject detections that are clearly not a person.
# Measured in pixels on the *original* (pre-letterbox) frame.
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

# --- Motion weighting (follow movers, ignore stationary leftovers) ---
# Each detected person is given a per-frame motion weight by matching to the
# nearest previous-frame detection. Cluster pull then uses summed motion
# weights instead of raw head-count, so:
#   * A lone person standing still after the action leaves → weight ~= floor,
#     so the camera does NOT lock onto them; it coasts and then SEARCHES.
#   * A moving group dominates even if a few static people are nearer.
#
# MOTION_WEIGHTING       — master switch. False = old count-based behaviour.
# MOTION_STATIC_FLOOR    — minimum weight a fully-static person contributes
#   (0.0 = static people are completely ignored, 1.0 = motion doesn't matter).
# MOTION_REF_SPEED       — frame-fractions per second at which a person hits
#   weight = 1.0. Walking ~0.10, running across frame ~0.30.
# MOTION_MATCH_DIST      — max distance (fraction of frame width) for matching
#   a current detection to a previous one when estimating per-person speed.
# MOTION_TOTAL_FLOOR     — if the SUM of motion weights across all detections
#   is below this, the tracker treats the scene as "no real action" and enters
#   coast/search instead of locking on whoever's left.
MOTION_WEIGHTING     = False   # reverted to v2 count-based clustering
MOTION_STATIC_FLOOR  = 0.10
MOTION_REF_SPEED     = 0.15
MOTION_MATCH_DIST    = 0.10
MOTION_TOTAL_FLOOR   = 0.35

# --- Directional consensus (anticipate a group break) ---
# When at least MOTION_CONSENSUS_FRAC of detections are actively moving
# (motion weight >= MOTION_MOVER_THRESHOLD), the static people are dropped
# from the target blend entirely and a lead offset is applied in the
# direction of the movers' average velocity. This is what lets the camera
# start panning as soon as ~half the players break, instead of waiting for
# the centroid of everyone (including stragglers) to drift.
#
# MOTION_MOVER_THRESHOLD — motion weight above which a person is "moving".
# MOTION_CONSENSUS_FRAC  — fraction of movers that triggers consensus mode.
# MOTION_LEAD_TIME       — seconds to project the action centre ahead when
#                          consensus is reached (scales with coherence).
# MOTION_COHERENCE_MIN   — minimum directional agreement (|Σv|/Σ|v|) needed
#                          before any lead is applied.
MOTION_MOVER_THRESHOLD = 0.45
MOTION_CONSENSUS_FRAC  = 0.30
MOTION_LEAD_TIME       = 1.0
MOTION_COHERENCE_MIN   = 0.30

# --- Pan/tilt coordination ---
PAN_PRIORITY_SCALE  = 1.2    # How strongly fast pan suppresses tilt EMA alpha

# --- Tilt-up bias (when person is very close) ---
TILT_BIAS_THRESHOLD = 0.9    # Person box height fraction to trigger tilt bias
TILT_UP_BIAS        = 1.0    # Max upward shift when overflowing
CLOSE_MAX_VEL       = 0.18   # Tilt speed cap when close

# --- Misc ---
COAST_SEC  = 3    # Keep moving N seconds after target disappears off screen (searching)

# ===================== COURT REGION-OF-INTEREST (v1, image-pixel) =====================
# Manual court calibration (see calibrate_court.py). When a court.json file
# exists in this folder it is loaded at startup and detections whose foot
# point falls outside the polygon are discarded. This stops the tracker from
# locking onto fans, passers-by, or anyone off the court.
#
# v1 LIMITATION — only usable when the camera can see the WHOLE court in one
# frame at its tracking pose. If the FOV is narrower than the court (typical
# outdoor setup), use PTZ position limits instead (see below) — that is the
# default for this project.
COURT_FILE            = Path(__file__).resolve().parent / "court.json"
COURT_FILTER_ENABLED  = False        # off by default; switch on if your venue fits
COURT_PADDING_PX      = 40
COURT_DRAW_OVERLAY    = True


# ===================== PTZ POSITION LIMITS (recommended) =====================
# Calibrate once with calibrate_ptz_limits.py: drive the camera with WASD/QE
# to each edge of the play area and tag the current pan/tilt/zoom as a min or
# max limit. At runtime, ptz_controller polls the camera's position via ONVIF
# GetStatus and refuses motion commands that would push it outside the box.
# This prevents the camera from panning at spectators / off the court even
# when its FOV is much smaller than the court itself.
#
# PTZ_LIMITS_FILE       — saved limits file. Auto-disabled if missing.
# PTZ_LIMITS_ENABLED    — master switch. False = no clamping (raw tracking).
# PTZ_LIMIT_SOFT_BAND   — within this distance of a limit, the velocity is
#   linearly scaled down to zero (smooth deceleration). Same units as the
#   normalized ONVIF position ([-1, 1] for pan/tilt, [0, 1] for zoom).
# PTZ_STATUS_HZ         — how often to poll GetStatus. 5 Hz is plenty.
PTZ_LIMITS_FILE       = Path(__file__).resolve().parent / "ptz_limits.json"
PTZ_LIMITS_ENABLED    = True
PTZ_LIMIT_SOFT_BAND   = 0.05
PTZ_STATUS_HZ         = 5.0

# Dead-reckoning fallback: some cameras (this Dahua included) do not report
# real PTZ position via ONVIF GetStatus — they return a constant value like
# (1.0, 1.0). When the controller detects this, it integrates the velocity
# commands it sends to estimate position internally. Drift is small over a
# calibration session (minutes); pressing H in the calibrator re-zeroes it.
#
# PTZ_DR_MODE  — "auto" (default, switch on when GetStatus looks broken)
#                "on"   (always integrate; ignore GetStatus position)
#                "off"  (never integrate; trust GetStatus only)
# PTZ_DR_SCALE — multiplier from ONVIF velocity*time to position units.
#                Pure 1:1 in normalized space. Adjust if your camera's
#                physical velocity differs from the normalized command.
PTZ_DR_MODE   = "auto"
PTZ_DR_SCALE  = 1.0

# Home preset: at startup main.py sends the camera to this preset number
# (defined in your camera's web UI) so that the dead-reckoned position
# origin matches the same physical pose every run. The calibrator's H key
# also uses it so the calibration origin and runtime origin are identical.
#
# How to set up:
#   1. In the camera's web UI → PTZ control, drive the camera to the pose
#      you want as "home" (typically: facing center court, mid-zoom).
#   2. In the Preset dropdown, select 1, click Add (추가) to save the
#      current position as preset 1.
#   3. Re-run calibrate_ptz_limits.py: pressing H now jumps to that pose.
#   4. Re-run main.py: same pose at startup.
#
# Set to None to disable preset homing (falls back to AbsoluteMove(0,0)).
PTZ_HOME_PRESET    = 1
PTZ_HOME_WAIT_SEC  = 4.0   # how long to wait for the preset move to finish

# Application
ENABLE_PTZ  = True
SHOW_WINDOW = True