"""
ptz_controller.py  —  ONVIF pan/tilt controller (no zoom for tracking).

One background thread sends ContinuousMove at 12.5 Hz and polls the camera's
current pan/tilt/zoom position via GetStatus at ~5 Hz. Main loop calls move()
or stop_move() — never blocks on network.

When PTZ position limits are loaded (see ptz_limits.json + calibrate_ptz_limits.py),
move() commands are soft-clamped: velocity is linearly scaled down to zero as
the camera approaches a limit, and hard-zeroed at the limit itself. This stops
the camera physically panning off the court.
"""
import json
import time
import threading
import logging
from onvif import ONVIFCamera
from config import (
    CAM_IP, CAM_PORT, CAM_USER, CAM_PASS, PT_MIN_VEL,
    PTZ_LIMITS_FILE, PTZ_LIMITS_ENABLED, PTZ_LIMIT_SOFT_BAND, PTZ_STATUS_HZ,
    PTZ_DR_MODE, PTZ_DR_SCALE,
)

_INTERVAL   = 0.08    # seconds between command sends (12.5 Hz)
_MIN_CHANGE = 0.03    # skip resend if velocity barely changed

log = logging.getLogger(__name__)


def _load_limits():
    """Return dict of limits or None when calibration is missing/disabled."""
    if not PTZ_LIMITS_ENABLED:
        return None
    if not PTZ_LIMITS_FILE.exists():
        log.info("No PTZ limits found at %s — clamping OFF. "
                 "Run calibrate_ptz_limits.py to enable.", PTZ_LIMITS_FILE)
        return None
    try:
        data = json.loads(PTZ_LIMITS_FILE.read_text())
        result = {
            "pan_min":  float(data.get("pan_min",  -1.0)),
            "pan_max":  float(data.get("pan_max",   1.0)),
            "tilt_min": float(data.get("tilt_min", -1.0)),
            "tilt_max": float(data.get("tilt_max",  1.0)),
        }
        log.info("PTZ limits loaded: pan [%.3f, %.3f]  tilt [%.3f, %.3f]",
                 result["pan_min"], result["pan_max"],
                 result["tilt_min"], result["tilt_max"])
        return result
    except Exception as e:
        log.warning("Failed to read ptz_limits.json (%s) — clamping OFF", e)
        return None


class PTZController:
    def __init__(self):
        self._ptz      = None
        self._token    = None
        self.connected = False

        self._lock     = threading.Lock()
        self._pan      = 0.0
        self._tilt     = 0.0
        self._active   = False
        self._running  = False

        # Position-state poll + limits
        self._limits         = _load_limits()
        self._position       = None    # (pan, tilt, zoom) from GetStatus, or None
        self._position_lock  = threading.Lock()
        self._last_status_t  = 0.0
        self._last_hit       = (False, False)  # (pan_hit, tilt_hit) for UI

        # Dead-reckoning state: integrate sent velocity to estimate position
        # when the camera doesn't report it via GetStatus.
        self._dr_pan        = 0.0
        self._dr_tilt       = 0.0
        self._dr_lock       = threading.Lock()
        self._dr_last_t     = None
        # Auto-detect broken GetStatus: count consecutive identical polls.
        self._status_last   = None
        self._status_same_n = 0
        # "on" forces DR from the start; "off" disables it; "auto" enables
        # only after we see GetStatus returning constant values.
        self._dr_active     = (PTZ_DR_MODE == "on")
        if self._dr_active:
            log.info("PTZ dead-reckoning: forced ON via PTZ_DR_MODE")

    # ── public ───────────────────────────────────────────────────────────────

    def connect(self):
        try:
            cam          = ONVIFCamera(CAM_IP, CAM_PORT, CAM_USER, CAM_PASS)
            media        = cam.create_media_service()
            self._ptz    = cam.create_ptz_service()
            profiles     = media.GetProfiles()
            self._token  = profiles[0].token
            self.connected = True
            log.info("ONVIF connected  token=%s", self._token)
            return True
        except Exception as e:
            log.error("ONVIF connect failed: %s", e)
            return False

    def start(self):
        """Start background command thread (call after connect)."""
        self._running = True
        threading.Thread(target=self._loop, daemon=True, name="PTZ").start()
        log.info("PTZ thread started")

    def move(self, pan, tilt):
        """Set desired pan/tilt velocity (non-blocking)."""
        with self._lock:
            self._pan, self._tilt, self._active = pan, tilt, True

    def stop_move(self):
        """Tell the thread to stop the motor (non-blocking)."""
        with self._lock:
            self._active = False

    def shutdown(self):
        self._running = False
        self._send_stop()

    def get_position(self):
        """Return last-known (pan, tilt, zoom) normalized, or None.
        Uses dead-reckoned position when GetStatus is broken/disabled."""
        if self._dr_active:
            with self._dr_lock:
                pan, tilt = self._dr_pan, self._dr_tilt
            with self._position_lock:
                zoom = self._position[2] if self._position is not None else 0.0
            return (pan, tilt, zoom)
        with self._position_lock:
            return self._position

    def reset_position(self, pan=0.0, tilt=0.0):
        """Force the internal dead-reckoned position to (pan, tilt).
        Useful after an AbsoluteMove home to clear accumulated drift."""
        with self._dr_lock:
            self._dr_pan, self._dr_tilt = pan, tilt
            self._dr_last_t = None

    def home_to_origin(self, wait_sec=3.0):
        """Send the camera to absolute (0,0) and reset dead-reckoning.

        Used at program startup so that the dead-reckoned position estimate
        matches the same physical origin that was used during calibration
        (the H key in calibrate_ptz_limits.py also sends AbsoluteMove(0,0)
        and resets DR).

        Returns True if AbsoluteMove was accepted, False otherwise. Even
        on failure the DR position is still zeroed so the user can manually
        re-center the camera and the limits will then be consistent.
        """
        if not self.connected:
            return False
        ok = False
        try:
            self._ptz.AbsoluteMove({
                "ProfileToken": self._token,
                "Position": {"PanTilt": {"x": 0.0, "y": 0.0}},
            })
            ok = True
            log.info("Homing camera to absolute (0,0) … waiting %.1fs",
                     wait_sec)
            time.sleep(max(0.0, wait_sec))
        except Exception as e:
            log.warning("AbsoluteMove home failed (%s) — "
                        "PTZ limits may not match physical position. "
                        "Manually centre the camera before tracking.", e)
        self.reset_position(0.0, 0.0)
        return ok

    def limits_active(self):
        return self._limits is not None

    def last_clamp_hit(self):
        """Return (pan_hit, tilt_hit) bools from the most recent move tick."""
        return self._last_hit

    # ── limit clamping ───────────────────────────────────────────────────────

    def _clamp_velocity(self, pan, tilt):
        """Soft-clamp velocity using current position vs configured limits.
        Returns (pan, tilt, hit_flags) where hit_flags is a 2-tuple of bools."""
        if self._limits is None:
            return pan, tilt, (False, False)
        with self._position_lock:
            pos = self._position
        if pos is None:
            return pan, tilt, (False, False)

        pan_now, tilt_now, _ = pos
        band = max(1e-6, PTZ_LIMIT_SOFT_BAND)
        pan_hit  = False
        tilt_hit = False

        if pan > 0:
            margin = self._limits["pan_max"] - pan_now
            if margin <= 0:
                pan = 0.0; pan_hit = True
            elif margin < band:
                pan *= margin / band; pan_hit = True
        elif pan < 0:
            margin = pan_now - self._limits["pan_min"]
            if margin <= 0:
                pan = 0.0; pan_hit = True
            elif margin < band:
                pan *= margin / band; pan_hit = True

        if tilt > 0:
            margin = self._limits["tilt_max"] - tilt_now
            if margin <= 0:
                tilt = 0.0; tilt_hit = True
            elif margin < band:
                tilt *= margin / band; tilt_hit = True
        elif tilt < 0:
            margin = tilt_now - self._limits["tilt_min"]
            if margin <= 0:
                tilt = 0.0; tilt_hit = True
            elif margin < band:
                tilt *= margin / band; tilt_hit = True

        return pan, tilt, (pan_hit, tilt_hit)

    # ── background thread ────────────────────────────────────────────────────

    def _poll_status_if_due(self, now):
        if PTZ_STATUS_HZ <= 0:
            return
        if now - self._last_status_t < 1.0 / PTZ_STATUS_HZ:
            return
        self._last_status_t = now
        try:
            s = self._ptz.GetStatus({"ProfileToken": self._token})
            pos = s.Position
            pan_now  = float(pos.PanTilt.x)
            tilt_now = float(pos.PanTilt.y)
            zoom_now = float(pos.Zoom.x) if pos.Zoom is not None else 0.0
            with self._position_lock:
                self._position = (pan_now, tilt_now, zoom_now)

            # Auto-detect broken GetStatus: if pan/tilt come back identical
            # for several polls in a row (e.g. stuck at 1.0), flip to DR.
            if PTZ_DR_MODE == "auto" and not self._dr_active:
                if self._status_last is not None and \
                        abs(pan_now  - self._status_last[0]) < 1e-4 and \
                        abs(tilt_now - self._status_last[1]) < 1e-4:
                    self._status_same_n += 1
                else:
                    self._status_same_n = 0
                self._status_last = (pan_now, tilt_now)
                if self._status_same_n >= 5:
                    self._dr_active = True
                    log.warning(
                        "GetStatus returns constant (%.3f, %.3f) over %d "
                        "polls — switching to dead reckoning.",
                        pan_now, tilt_now, self._status_same_n + 1)
        except Exception as e:
            log.debug("GetStatus failed: %s", e)
            if PTZ_DR_MODE == "auto" and not self._dr_active:
                self._status_same_n += 1
                if self._status_same_n >= 5:
                    self._dr_active = True
                    log.warning("GetStatus repeatedly failed — "
                                "switching to dead reckoning.")

    def _loop(self):
        last_pan   = None
        last_tilt  = None
        was_active = False

        while self._running:
            time.sleep(_INTERVAL)
            if not self.connected:
                continue

            now = time.monotonic()
            self._poll_status_if_due(now)

            with self._lock:
                pan, tilt, active = self._pan, self._tilt, self._active

            if active:
                pan, tilt, hit = self._clamp_velocity(pan, tilt)
                self._last_hit = hit

                if abs(pan)  < PT_MIN_VEL: pan  = 0.0
                if abs(tilt) < PT_MIN_VEL: tilt = 0.0

                # Dead-reckoning: integrate the velocity we're about to send.
                if PTZ_DR_MODE != "off":
                    with self._dr_lock:
                        if self._dr_last_t is not None:
                            dt = now - self._dr_last_t
                            if 0.0 < dt < 1.0:
                                self._dr_pan  = max(-1.0, min(1.0,
                                    self._dr_pan  + pan  * dt * PTZ_DR_SCALE))
                                self._dr_tilt = max(-1.0, min(1.0,
                                    self._dr_tilt + tilt * dt * PTZ_DR_SCALE))
                        self._dr_last_t = now

                if pan == 0.0 and tilt == 0.0:
                    if was_active:
                        self._send_stop()
                        was_active = False
                        last_pan = last_tilt = None
                    continue

                changed = (last_pan is None
                           or abs(pan  - last_pan)  > _MIN_CHANGE
                           or abs(tilt - last_tilt) > _MIN_CHANGE)
                if changed:
                    try:
                        self._ptz.ContinuousMove({
                            "ProfileToken": self._token,
                            "Velocity": {"PanTilt": {"x": pan, "y": tilt}},
                        })
                        last_pan, last_tilt = pan, tilt
                        was_active = True
                    except Exception as e:
                        log.error("ContinuousMove failed: %s", e)
                        self.connected = False
            elif was_active:
                self._send_stop()
                was_active = False
                last_pan = last_tilt = None
                self._last_hit = (False, False)
                with self._dr_lock:
                    self._dr_last_t = None

    def _send_stop(self):
        if not self.connected:
            return
        try:
            self._ptz.Stop({
                "ProfileToken": self._token,
                "PanTilt": True,
                "Zoom":    False,
            })
        except Exception as e:
            log.error("Stop failed: %s", e)
