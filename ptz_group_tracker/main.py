"""
main.py – Group PTZ tracker for sports courts (basketball, football, etc.)

Pipeline per frame
──────────────────
  1. Grab latest frame from RTSP (background thread, zero-latency).
  2. Detect all people (+ ball if TRACK_BALL=True) with YOLO.
  3. Compute group envelope bounding box around all detections.
  4. GroupTracker P-controller computes pan/tilt velocity + zoom error.
  5. Non-blocking: set_pantilt() / set_zoom_error() hand values to PTZ threads.
  6. Pan/tilt thread sends ContinuousMove at ~12 Hz (only when changed).
  7. Zoom thread sends ContinuousMove (zoom-only) every 0.35 s independently.

This pipeline never blocks on network I/O — YOLO inference is the only
rate-limiting factor (~15–25 fps on CPU at 416 px).

Press Q to quit.
"""

import cv2
import os
import sys
import subprocess
import time
import logging
import threading
from collections import deque

from config import (
    RTSP_URL,
    SHOW_DEBUG_WINDOW,
    CAM_IP, ENABLE_PTZ, ENABLE_AUDIO,
)
from detector       import PeopleDetector
from tracker        import GroupTracker
from ptz_controller import PTZController

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|"
    "analyzeduration;0|"
    "probesize;32768|"
    "fflags;nobuffer|"
    "flags;low_delay|"
    "max_delay;0|"
    "allowed_media_types;video"
)

# ── Silence FFmpeg AU header spam ────────────────────────────────────────────
# FFmpeg writes directly to the Windows STDERR handle (C-level, not Python).
# The correct Windows fix is SetStdHandle(-12, nul) via kernel32.
# os.dup2 does not work reliably on Windows for fd 2 — skip it.
import ctypes as _ctypes
try:
    _k32  = _ctypes.windll.kernel32
    _nul  = _k32.CreateFileW("nul", 0x40000000, 3, None, 3, 0, None)
    if _nul and _nul != -1:
        _k32.SetStdHandle(-12, _nul)   # STD_ERROR_HANDLE = -12
except Exception:
    pass

# Redirect Python's stderr wrapper so any Python-level stderr also goes nowhere.
sys.excepthook = lambda t, v, tb: __import__('traceback').print_exception(t, v, tb, file=sys.__stdout__)
sys.stderr = open(os.devnull, 'w')

# Force stdout to utf-8 so log messages with non-ASCII chars don't crash on
# systems with a non-utf-8 locale (e.g. Windows cp949).
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

_log_handler = logging.StreamHandler(sys.stdout)
_log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[_log_handler])
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Low-latency RTSP reader
# ══════════════════════════════════════════════════════════════════════════════

class RTSPStream:
    """Background thread that continuously grabs the latest frame."""

    def __init__(self, url: str, reconnect_delay: float = 3.0) -> None:
        self._url             = url
        self._reconnect_delay = reconnect_delay
        self._frame           = None
        self._lock            = threading.Lock()
        self._running         = False
        self._thread          = None

    def start(self) -> "RTSPStream":
        self._running = True
        self._thread  = threading.Thread(
            target=self._capture_loop, daemon=True, name="RTSPStream"
        )
        self._thread.start()
        return self

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _capture_loop(self) -> None:
        while self._running:
            logger.info("Opening RTSP: %s", self._url)
            cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                logger.warning("Cannot open RTSP – retrying in %.0fs …", self._reconnect_delay)
                time.sleep(self._reconnect_delay)
                continue
            logger.info("RTSP stream opened.")
            while self._running:
                # cap.read() blocks until the next frame arrives from the camera.
                # Because this runs in its own thread (not the YOLO thread),
                # frames are consumed as fast as the camera sends them.
                # CAP_PROP_BUFFERSIZE=1 + nobuffer flags keep only one frame
                # queued, so self._frame is always the live image.
                ret, frame = cap.read()
                if not ret:
                    logger.warning("RTSP read failed – reconnecting …")
                    break
                with self._lock:
                    self._frame = frame
            cap.release()
            if self._running:
                time.sleep(self._reconnect_delay)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    stream     = RTSPStream(RTSP_URL, 3.0).start()
    detector   = PeopleDetector()
    tracker    = GroupTracker()
    controller = PTZController()

    # Audio: launch ffplay as a background subprocess (audio-only, no window).
    # ffplay ships with FFmpeg so no extra install is needed.
    audio_proc = None
    if ENABLE_AUDIO:
        try:
            audio_proc = subprocess.Popen(
                [
                    "ffplay",
                    "-rtsp_transport", "tcp",
                    "-i", RTSP_URL,
                    "-vn",          # video disabled
                    "-nodisp",      # no window
                    "-loglevel", "quiet",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("Audio playback started (ffplay pid %d).", audio_proc.pid)
        except FileNotFoundError:
            logger.warning("ffplay not found — audio disabled. Install FFmpeg to enable audio.")
            audio_proc = None

    if ENABLE_PTZ:
        logger.info("Connecting to ONVIF at %s …", CAM_IP)
        controller.connect()

    _fps_times:   deque = deque(maxlen=30)
    _last_diag:   float = 0.0
    _frame_idx:   int   = 0          # counts every main-loop iteration
    _DETECT_EVERY: int  = 2          # run YOLO every N frames (1 = every frame, 2 = every other)
    # Cache last detection results so non-YOLO frames still draw correctly
    _detections = {"persons": [], "ball": None}
    _pan_vel    = None
    _tilt_vel   = None
    _group_box  = None

    logger.info("Waiting for first frame …")
    while stream.read() is None:
        time.sleep(0.1)
    logger.info("Stream ready.")

    # ── Startup: zoom all the way out so we start from a known wide view ──────
    if ENABLE_PTZ and controller.is_connected:
        logger.info("Startup: zooming out for 3 s …")
        controller.zoom_all_out(duration=3.0)
        logger.info("Zoom reset done. Starting PTZ threads.")
        controller.start_threads()

    # ── Main detection + control loop ─────────────────────────────────────────
    while True:
        frame = stream.read()
        if frame is None:
            time.sleep(0.05)
            continue

        _fps_times.append(time.monotonic())
        _frame_idx += 1
        frame_h, frame_w = frame.shape[:2]

        # Run YOLO only every _DETECT_EVERY frames to reduce CPU load.
        # Between detections the camera commands keep running from cached values.
        if _frame_idx % _DETECT_EVERY == 0:
            _detections = detector.detect(frame)
            _pan_vel, _tilt_vel, _group_box = tracker.update(
                _detections, frame_w, frame_h
            )

            # Send commands (non-blocking — stored for background thread)
            if ENABLE_PTZ and controller.is_connected:
                if _pan_vel is None:
                    controller.clear_pantilt()
                else:
                    controller.set_pantilt(_pan_vel, _tilt_vel)

        pan_vel   = _pan_vel
        tilt_vel  = _tilt_vel
        group_box = _group_box
        detections = _detections

        # Periodic console diagnostics every 2 s so we can see what's happening
        now = time.monotonic()
        if now - _last_diag >= 2.0:
            _last_diag = now
            fps = (len(_fps_times) - 1) / (_fps_times[-1] - _fps_times[0]) if len(_fps_times) >= 2 else 0
            logger.info(
                "FPS=%.1f  people=%d  pan_e=%.3f  tilt_e=%.3f  vel=%s",
                fps,
                len(detections["persons"]),
                tracker.pan_ema,
                tracker.tilt_ema,
                f"{pan_vel:.2f}" if pan_vel is not None else "STOP",
            )

        # ── Debug overlay ─────────────────────────────────────────────────────
        if SHOW_DEBUG_WINDOW:
            s = 1.0
            disp_w, disp_h = frame_w, frame_h
            vis = frame.copy()

            # Individual person boxes — green
            for x1, y1, x2, y2 in detections["persons"]:
                cv2.rectangle(vis,
                              (int(x1*s), int(y1*s)), (int(x2*s), int(y2*s)),
                              (0, 210, 0), 1)

            # Ball box — orange
            if detections["ball"] is not None:
                bx1, by1, bx2, by2 = detections["ball"]
                cv2.rectangle(vis,
                              (int(bx1*s), int(by1*s)), (int(bx2*s), int(by2*s)),
                              (0, 140, 255), 2)

            # Group envelope — cyan
            if group_box is not None:
                gx1, gy1, gx2, gy2 = group_box
                cv2.rectangle(vis,
                              (int(gx1*s), int(gy1*s)), (int(gx2*s), int(gy2*s)),
                              (0, 255, 255), 2)
                gcx = int((gx1 + gx2) * 0.5 * s)
                gcy = int((gy1 + gy2) * 0.5 * s)
                cv2.drawMarker(vis, (gcx, gcy), (0, 255, 255), cv2.MARKER_CROSS, 28, 2)

            # Frame centre
            cv2.drawMarker(vis, (disp_w // 2, disp_h // 2),
                           (0, 255, 0), cv2.MARKER_CROSS, 20, 1)

            # HUD
            fps      = 0.0
            if len(_fps_times) >= 2:
                fps = (len(_fps_times) - 1) / (_fps_times[-1] - _fps_times[0])
            n_people = len(detections["persons"])
            has_ball = detections["ball"] is not None
            state    = "COASTING" if (group_box is None and pan_vel is not None) else ("TRACKING" if group_box is not None else "SEARCHING")
            s_col    = (0, 165, 255) if state == "COASTING" else ((0, 255, 255) if state == "TRACKING" else (0, 80, 255))

            cv2.putText(vis, f"FPS {fps:.1f}  {frame_w}x{frame_h}",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(vis, state,
                        (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.80, s_col, 2, cv2.LINE_AA)
            cv2.putText(vis, f"People: {n_people}{'   Ball detected' if has_ball else ''}",
                        (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(vis, f"Pan {tracker.pan_ema:+.2f}  Tilt {tracker.tilt_ema:+.2f}",
                        (10, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 180, 0), 1, cv2.LINE_AA)
            vel_txt = f"Vel pan={pan_vel:.2f} tilt={tilt_vel:.2f}" if pan_vel is not None else "Vel STOP"
            cv2.putText(vis, vel_txt,
                        (10, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 0), 1, cv2.LINE_AA)
            onvif_lbl = "ONVIF OK" if controller.is_connected else "ONVIF OFFLINE"
            onvif_col = (0, 200, 0)  if controller.is_connected else (0, 60, 220)
            cv2.putText(vis, onvif_lbl,
                        (10, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.45, onvif_col, 1, cv2.LINE_AA)

            cv2.imshow("Group Tracker", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # ── Shutdown ──────────────────────────────────────────────────────────────
    if ENABLE_PTZ:
        controller.shutdown()
    stream.stop()
    if audio_proc is not None:
        audio_proc.terminate()
        logger.info("Audio playback stopped.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
