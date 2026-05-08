"""
ptz_controller.py  —  ONVIF pan/tilt controller (no zoom).

One background thread sends ContinuousMove at 12.5 Hz.
Main loop calls move() or stop_move() — never blocks on network.
"""
import time
import threading
import logging
from onvif import ONVIFCamera
from config import CAM_IP, CAM_PORT, CAM_USER, CAM_PASS, PT_MIN_VEL

_INTERVAL   = 0.08    # seconds between commands (12.5 Hz)
_MIN_CHANGE = 0.03    # skip resend if velocity barely changed

log = logging.getLogger(__name__)


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

    # ── Background thread ────────────────────────────────────────────────────

    def _loop(self):
        last_pan   = None
        last_tilt  = None
        was_active = False

        while self._running:
            time.sleep(_INTERVAL)
            if not self.connected:
                continue

            with self._lock:
                pan, tilt, active = self._pan, self._tilt, self._active

            if active:
                # clamp below noise floor
                if abs(pan)  < PT_MIN_VEL: pan  = 0.0
                if abs(tilt) < PT_MIN_VEL: tilt = 0.0

                if pan == 0.0 and tilt == 0.0:
                    if was_active:
                        self._send_stop()
                        was_active = last_pan = last_tilt = None
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
                was_active = last_pan = last_tilt = None

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