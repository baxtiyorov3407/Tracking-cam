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
  [  /  ]    decrease / increase manual-drive speed (shown in HUD)
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

# Manual-drive speeds (normalized ONVIF velocity, [-1, 1]).
# Defaults are intentionally slow so a single tap moves only ~1 degree.
# Use [ and ] in the tool to scale live.
JOG_PAN_BASE     = 0.08
JOG_TILT_BASE    = 0.08
JOG_ZOOM_BASE    = 0.10
KEY_TIMEOUT_SEC  = 0.08    # shorter -> single tap = short move

SPEED_MIN        = 0.10
SPEED_MAX        = 4.00
SPEED_STEP       = 1.25    # multiplier per [ / ] press

# When the camera is zoomed in, a given normalized velocity rotates the view
# the same angular amount BUT the apparent on-screen movement is much larger.
# Scaling pan/tilt by (1 - zoom * ZOOM_SLOWDOWN) keeps the manual feel
# consistent: 0.0 = no scaling, 0.9 = pan is 10x slower at full zoom.
ZOOM_SLOWDOWN    = 0.85


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


def draw_hud(frame, ptz, limits, speed_mult, status_msg=None, status_ok=True):
    h, w = frame.shape[:2]
    pos  = ptz.get_position()
    vis  = frame.copy()

    cv2.rectangle(vis, (0, 0), (w, 110), (0, 0, 0), -1)
    cv2.putText(vis, "PTZ LIMITS CALIBRATION", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, "WASD pan/tilt   Q/E zoom   [ ] speed   1 LEFT 2 RIGHT 3 TOP 4 BOTTOM",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(vis, "H home   X clear   ENTER save   ESC quit",
                (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    if pos is None:
        cv2.putText(vis, "GetStatus: waiting…", (10, 96),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 255), 1, cv2.LINE_AA)
    else:
        p, t, z = pos
        cv2.putText(vis,
                    f"now  pan={p:+.3f}  tilt={t:+.3f}  zoom={z:.3f}    speed={speed_mult:.2f}x",
                    (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    # Bottom-screen status banner (shown after save attempt etc.)
    if status_msg:
        col = (0, 200, 0) if status_ok else (0, 80, 255)
        cv2.rectangle(vis, (0, h - 36), (w, h), (0, 0, 0), -1)
        cv2.putText(vis, status_msg, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

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
        msg = f"Cannot save — missing: {', '.join(missing)}"
        print(msg)
        return False, msg
    # The 1/2/3/4 labels are just mnemonics; the camera's axis sign may be
    # reversed from "left = lower number". Auto-sort so order of capture
    # never matters.
    pmn, pmx = sorted((limits["pan_min"],  limits["pan_max"]))
    tmn, tmx = sorted((limits["tilt_min"], limits["tilt_max"]))
    if pmn == pmx or tmn == tmx:
        msg = "Cannot save — pan or tilt limits are identical. Re-capture."
        print(msg)
        return False, msg
    out = {"pan_min": pmn, "pan_max": pmx, "tilt_min": tmn, "tilt_max": tmx}
    out["camera_ip"] = CAM_IP
    out["saved_at"]  = time.strftime("%Y-%m-%d %H:%M:%S")
    OUT_FILE.write_text(json.dumps(out, indent=2))
    msg = f"Saved limits to {OUT_FILE}"
    print(msg)
    return True, msg


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
    speed_mult = 1.0
    status_msg = None
    status_ok  = True
    status_until = 0.0

    print("Ready. Use WASD to pan/tilt, Q/E to zoom, [ ] to change speed, "
          "1-4 to capture limits, ENTER to save.")

    while True:
        frame = stream.read()
        if frame is None:
            time.sleep(0.02)
            continue
        # Clear stale status banners after 4 seconds
        msg = status_msg if time.monotonic() < status_until else None
        cv2.imshow(win, draw_hud(frame, ptz, limits, speed_mult, msg, status_ok))
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

        # Current effective pan/tilt jog speed, scaled by user multiplier
        # AND by zoom (so it doesn't feel violent when zoomed in).
        pos_now = ptz.get_position()
        zoom_now = pos_now[2] if pos_now is not None else 0.0
        zoom_scale = max(0.05, 1.0 - ZOOM_SLOWDOWN * zoom_now)
        pan_sp  = JOG_PAN_BASE  * speed_mult * zoom_scale
        tilt_sp = JOG_TILT_BASE * speed_mult * zoom_scale
        zoom_sp = JOG_ZOOM_BASE * speed_mult     # zoom not scaled by zoom

        # Motion keys
        if key == ord('w'):
            ptz.move(0.0,  tilt_sp); moving_pt = True
        elif key == ord('s'):
            ptz.move(0.0, -tilt_sp); moving_pt = True
        elif key == ord('a'):
            ptz.move(-pan_sp, 0.0);  moving_pt = True
        elif key == ord('d'):
            ptz.move( pan_sp, 0.0);  moving_pt = True
        elif key == ord('q'):
            _zoom_continuous(ptz, -zoom_sp); moving_zm = True
        elif key == ord('e'):
            _zoom_continuous(ptz,  zoom_sp); moving_zm = True
        elif key == ord('['):
            speed_mult = max(SPEED_MIN, speed_mult / SPEED_STEP)
            print(f"speed = {speed_mult:.2f}x")
        elif key == ord(']'):
            speed_mult = min(SPEED_MAX, speed_mult * SPEED_STEP)
            print(f"speed = {speed_mult:.2f}x")
        elif key == ord('h'):
            _absolute_home(ptz)
        elif key == ord('x'):
            limits.clear()
            status_msg, status_ok, status_until = (
                "Cleared all captured limits.", True, now + 3.0)
            print(status_msg)
        elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
            pos = ptz.get_position()
            if pos is None:
                status_msg = "PTZ position not available yet (GetStatus pending)."
                status_ok, status_until = False, now + 3.0
                print(status_msg)
            else:
                p, t, _ = pos
                if key == ord('1'):
                    limits["pan_min"]  = p
                    status_msg = f"Captured LEFT  pan_min  = {p:+.3f}"
                elif key == ord('2'):
                    limits["pan_max"]  = p
                    status_msg = f"Captured RIGHT pan_max  = {p:+.3f}"
                elif key == ord('3'):
                    limits["tilt_max"] = t
                    status_msg = f"Captured TOP   tilt_max = {t:+.3f}"
                elif key == ord('4'):
                    limits["tilt_min"] = t
                    status_msg = f"Captured BOTTOM tilt_min = {t:+.3f}"
                status_ok, status_until = True, now + 3.0
                print(status_msg)
        elif key in (13, 10):                        # ENTER (CR or LF)
            ok, msg = save(limits)
            status_msg, status_ok, status_until = msg, ok, now + 5.0
            if ok:
                # Render one final frame so the user sees the success banner,
                # then exit.
                cv2.imshow(win, draw_hud(frame, ptz, limits, speed_mult, msg, True))
                cv2.waitKey(800)
                break

    # Cleanup
    ptz.stop_move()
    _zoom_stop(ptz)
    ptz.shutdown()
    stream.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
