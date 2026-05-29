"""
calibrate_ptz_limits.py  —  Interactive PTZ position-limit calibration.

Drive the camera with WASD (pan/tilt) and Q/E (zoom) until it is pointing at
one extreme of the play area, then press the matching digit to record the
current pan/tilt as a limit. Save with ENTER, quit with ESC.

The saved ptz_limits.json is loaded by ptz_controller at startup; any tracking
command that would push the camera past these limits is soft-clamped to zero.

Controls
  W / S      tilt up / down
  A / D      pan  left / right
  Q / E      zoom out / in
  hold key   keep moving;  release  =  stop
  1          set LEFT  pan limit at current position
  2          set RIGHT pan limit
  3          set TOP   tilt limit
  4          set BOTTOM tilt limit
  H          send the camera back to (pan=0, tilt=0) home
  X          clear all currently captured limits
  ENTER      save ptz_limits.json and exit  (all four limits must be set)
  ESC / `    quit without saving
"""
import os, sys, json, time, threading
from pathlib import Path

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|stimeout;5000000|fflags;nobuffer|flags;low_delay|"
    "analyzeduration;0|probesize;32768|max_delay;0"
)
os.environ.setdefault("OPENCV_LOG_LEVEL",      "SILENT")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2

from config         import RTSP_URL, CAM_IP
from ptz_controller import PTZController

OUT_FILE = Path(__file__).resolve().parent / "ptz_limits.json"

# Manual-drive speeds (normalized ONVIF velocity, [-1, 1])
JOG_PAN_SPEED   = 0.30
JOG_TILT_SPEED  = 0.25
JOG_ZOOM_SPEED  = 0.25
KEY_TIMEOUT_SEC = 0.15   # release inferred when no key seen this long


class StreamReader:
    """Latest-frame RTSP reader (subset of main.py's RTSPStream)."""
    def __init__(self, url):
        self._url    = url
        self._frame  = None
        self._lock   = threading.Lock()
        self._stop   = False
        threading.Thread(target=self._loop, daemon=True).start()

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._stop = True

    def _loop(self):
        cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while not self._stop:
            ret, f = cap.read()
            if not ret or f is None:
                cap.release()
                time.sleep(0.3)
                cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue
            with self._lock:
                self._frame = f
        cap.release()


def _zoom_continuous(ptz, vz):
    """Send a zoom-only ContinuousMove using the underlying ONVIF service."""
    try:
        ptz._ptz.ContinuousMove({
            "ProfileToken": ptz._token,
            "Velocity": {"Zoom": {"x": vz}},
        })
    except Exception as e:
        print(f"zoom move failed: {e}")


def _zoom_stop(ptz):
    try:
        ptz._ptz.Stop({"ProfileToken": ptz._token,
                       "PanTilt": False, "Zoom": True})
    except Exception:
        pass


def _absolute_home(ptz):
    """Send camera to (0,0). Not all cameras implement AbsoluteMove cleanly;
    skip silently on error — user can drive manually with WASD."""
    try:
        ptz._ptz.AbsoluteMove({
            "ProfileToken": ptz._token,
            "Position": {"PanTilt": {"x": 0.0, "y": 0.0}},
        })
    except Exception as e:
        print(f"home (AbsoluteMove) not supported by camera: {e}")


def draw_hud(frame, ptz, limits):
    h, w = frame.shape[:2]
    pos  = ptz.get_position()
    vis  = frame.copy()

    cv2.rectangle(vis, (0, 0), (w, 110), (0, 0, 0), -1)
    cv2.putText(vis, "PTZ LIMITS CALIBRATION", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, "WASD pan/tilt   Q/E zoom   1 LEFT  2 RIGHT  3 TOP  4 BOTTOM",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(vis, "H home   X clear   ENTER save   ESC quit",
                (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    if pos is None:
        cv2.putText(vis, "GetStatus: waiting…", (10, 96),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 255), 1, cv2.LINE_AA)
    else:
        p, t, z = pos
        cv2.putText(vis,
                    f"now  pan={p:+.3f}  tilt={t:+.3f}  zoom={z:.3f}",
                    (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    # Right-hand panel: captured limits
    def line(label, v, y, set_):
        col = (0, 255, 0) if set_ else (80, 80, 80)
        txt = f"{label}: {v:+.3f}" if set_ else f"{label}: (not set)"
        cv2.putText(vis, txt, (w - 260, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)

    line("1 LEFT  ",  limits.get("pan_min",  0), 24,  "pan_min"  in limits)
    line("2 RIGHT ",  limits.get("pan_max",  0), 48,  "pan_max"  in limits)
    line("3 TOP   ",  limits.get("tilt_max", 0), 72,  "tilt_max" in limits)
    line("4 BOTTOM", limits.get("tilt_min",  0), 96,  "tilt_min" in limits)

    return vis


def save(limits):
    needed = ["pan_min", "pan_max", "tilt_min", "tilt_max"]
    missing = [k for k in needed if k not in limits]
    if missing:
        print(f"Missing limits: {missing}. Set them all before saving.")
        return False
    if limits["pan_min"] >= limits["pan_max"]:
        print("pan_min must be < pan_max. Re-capture LEFT/RIGHT.")
        return False
    if limits["tilt_min"] >= limits["tilt_max"]:
        print("tilt_min must be < tilt_max. Re-capture TOP/BOTTOM.")
        return False
    out = dict(limits)
    out["camera_ip"] = CAM_IP
    out["saved_at"]  = time.strftime("%Y-%m-%d %H:%M:%S")
    OUT_FILE.write_text(json.dumps(out, indent=2))
    print(f"Saved limits to {OUT_FILE}")
    return True


def main():
    print(f"Connecting to camera at {CAM_IP} …")
    ptz = PTZController()
    if not ptz.connect():
        print("ONVIF connection failed; cannot calibrate. Check config.py.")
        sys.exit(1)
    ptz.start()

    stream = StreamReader(RTSP_URL)

    win = "PTZ Limits Calibration"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    limits: dict = {}
    last_key_t = 0.0
    moving_pt  = False
    moving_zm  = False

    print("Ready. Use WASD to pan/tilt, Q/E to zoom, 1-4 to capture limits.")

    while True:
        frame = stream.read()
        if frame is None:
            time.sleep(0.02)
            continue
        cv2.imshow(win, draw_hud(frame, ptz, limits))
        key = cv2.waitKey(15) & 0xFF
        now = time.monotonic()

        # No key seen — stop any in-progress motion after a brief debounce.
        if key == 0xFF:
            if moving_pt and now - last_key_t > KEY_TIMEOUT_SEC:
                ptz.stop_move(); moving_pt = False
            if moving_zm and now - last_key_t > KEY_TIMEOUT_SEC:
                _zoom_stop(ptz); moving_zm = False
            continue

        last_key_t = now

        if key in (27, ord('`')):                    # ESC / `
            print("Quit without saving.")
            break

        # Motion keys
        if key == ord('w'):
            ptz.move(0.0,  JOG_TILT_SPEED); moving_pt = True
        elif key == ord('s'):
            ptz.move(0.0, -JOG_TILT_SPEED); moving_pt = True
        elif key == ord('a'):
            ptz.move(-JOG_PAN_SPEED, 0.0);  moving_pt = True
        elif key == ord('d'):
            ptz.move( JOG_PAN_SPEED, 0.0);  moving_pt = True
        elif key == ord('q'):
            _zoom_continuous(ptz, -JOG_ZOOM_SPEED); moving_zm = True
        elif key == ord('e'):
            _zoom_continuous(ptz,  JOG_ZOOM_SPEED); moving_zm = True
        elif key == ord('h'):
            _absolute_home(ptz)
        elif key == ord('x'):
            limits.clear()
            print("Cleared all captured limits.")
        elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
            pos = ptz.get_position()
            if pos is None:
                print("PTZ position not available yet (GetStatus pending).")
            else:
                p, t, _ = pos
                if key == ord('1'):
                    limits["pan_min"]  = p; print(f"LEFT pan_min  = {p:+.3f}")
                elif key == ord('2'):
                    limits["pan_max"]  = p; print(f"RIGHT pan_max = {p:+.3f}")
                elif key == ord('3'):
                    limits["tilt_max"] = t; print(f"TOP tilt_max  = {t:+.3f}")
                elif key == ord('4'):
                    limits["tilt_min"] = t; print(f"BOTTOM tilt_min = {t:+.3f}")
        elif key == 13:                              # ENTER
            if save(limits):
                break

    # Cleanup
    ptz.stop_move()
    _zoom_stop(ptz)
    ptz.shutdown()
    stream.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
