"""
ptz_controller.py - Threaded ONVIF PTZ controller (pan/tilt only, no zoom).

One background thread fires ContinuousMove(PanTilt) at a fixed interval.
Main loop calls set_pantilt() or clear_pantilt() — never blocks on network I/O.
"""

import time
import threading
import logging
import numpy as np
from onvif import ONVIFCamera
from config import CAM_IP, CAM_PORT, CAM_USER, CAM_PASS, PT_MIN_VEL

_SEND_INTERVAL = 0.05   # seconds between pan/tilt ContinuousMove commands (20 Hz)
_MIN_CHANGE    = 0.04   # skip resend if velocity unchanged by less than this

logger = logging.getLogger(__name__)


class PTZController:

    def __init__(self) -> None:
        self._ptz       = None
        self._token     = None
        self._connected = False

        self._pt_lock      = threading.Lock()
        self._desired_pan  = 0.0
        self._desired_tilt = 0.0
        self._pt_active    = False

        self._running   = False
        self._pt_thread = None

    # -- Connection -----------------------------------------------------------

    def connect(self) -> bool:
        try:
            cam   = ONVIFCamera(CAM_IP, CAM_PORT, CAM_USER, CAM_PASS)
            media = cam.create_media_service()
            self._ptz = cam.create_ptz_service()
            profiles  = media.GetProfiles()
            if not profiles:
                logger.error("ONVIF: no profiles found.")
                return False
            self._token     = profiles[0].token
            self._connected = True
            logger.info("ONVIF connected. Token: %s", self._token)
            return True
        except Exception as exc:
            logger.error("ONVIF connect failed: %s", exc)
            self._connected = False
            return False

    # -- Thread management ----------------------------------------------------

    def start_threads(self) -> None:
        """Start the pan/tilt background thread (call after connect)."""
        self._running   = True
        self._pt_thread = threading.Thread(
            target=self._pantilt_loop, daemon=True, name="PTZ-PanTilt")
        self._pt_thread.start()
        logger.info("PTZ pan/tilt thread started.")

    # -- Public setters (non-blocking) ----------------------------------------

    def set_pantilt(self, pan: float, tilt: float) -> None:
        with self._pt_lock:
            self._desired_pan  = pan
            self._desired_tilt = tilt
            self._pt_active    = True

    def clear_pantilt(self) -> None:
        with self._pt_lock:
            self._pt_active = False

    # -- Startup helper -------------------------------------------------------

    def zoom_all_out(self, duration: float = 3.0) -> None:
        """Blocking: zoom to widest angle for `duration` seconds at startup."""
        if not self._connected:
            return
        try:
            self._ptz.ContinuousMove({
                "ProfileToken": self._token,
                "Velocity": {
                    "PanTilt": {"x": 0.0, "y": 0.0},
                    "Zoom":    {"x": -1.0},
                },
            })
            time.sleep(duration)
            self.stop_all()
        except Exception as exc:
            logger.warning("zoom_all_out failed: %s", exc)

    # -- Background thread ----------------------------------------------------

    def _pantilt_loop(self) -> None:
        _last_pan   = None
        _last_tilt  = None
        _was_active = False

        while self._running:
            time.sleep(_SEND_INTERVAL)
            if not self._connected:
                continue

            with self._pt_lock:
                pan    = self._desired_pan
                tilt   = self._desired_tilt
                active = self._pt_active

            if active:
                # Clamp tiny velocities to zero — avoids slow motor crawl/noise
                if abs(pan)  < PT_MIN_VEL: pan  = 0.0
                if abs(tilt) < PT_MIN_VEL: tilt = 0.0

                if pan == 0.0 and tilt == 0.0:
                    if _was_active:
                        try:
                            self._ptz.Stop({
                                "ProfileToken": self._token,
                                "PanTilt": True, "Zoom": False,
                            })
                            _was_active = False
                            _last_pan   = None
                            _last_tilt  = None
                        except Exception as exc:
                            logger.error("PTZ Stop failed: %s", exc)
                    continue

                changed = (
                    _last_pan is None or
                    abs(pan  - _last_pan)  > _MIN_CHANGE or
                    abs(tilt - _last_tilt) > _MIN_CHANGE
                )
                if changed:
                    try:
                        self._ptz.ContinuousMove({
                            "ProfileToken": self._token,
                            "Velocity": {
                                "PanTilt": {"x": pan, "y": tilt},
                            },
                        })
                        _last_pan   = pan
                        _last_tilt  = tilt
                        _was_active = True
                    except Exception as exc:
                        logger.error("ContinuousMove failed: %s", exc)
                        self._connected = False
            elif _was_active:
                try:
                    self._ptz.Stop({
                        "ProfileToken": self._token,
                        "PanTilt": True,
                        "Zoom":    False,
                    })
                    _was_active = False
                    _last_pan   = None
                    _last_tilt  = None
                except Exception as exc:
                    logger.error("PTZ Stop failed: %s", exc)

    # -- Helpers --------------------------------------------------------------

    def stop_all(self) -> None:
        if not self._connected:
            return
        try:
            self._ptz.Stop({
                "ProfileToken": self._token,
                "PanTilt": True,
                "Zoom":    True,
            })
        except Exception as exc:
            logger.error("stop_all failed: %s", exc)

    def shutdown(self) -> None:
        self._running = False
        self.stop_all()

    @property
    def is_connected(self) -> bool:
        return self._connected