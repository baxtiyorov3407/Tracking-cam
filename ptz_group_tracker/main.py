"""
main.py  —  PTZ Basketball Group Tracker
Press Q to quit.
"""
# ── FFMPEG options MUST be set before cv2 is imported ────────────────────────
import os, sys
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|stimeout;5000000|fflags;nobuffer|flags;low_delay|"
    "analyzeduration;0|probesize;32768|max_delay;0"
)
os.environ.setdefault("OPENCV_LOG_LEVEL",      "SILENT")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2, ctypes, time, threading, logging
from collections import deque

# Silence FFmpeg C-level stderr on Windows
try:
    _k32 = ctypes.windll.kernel32
    _nul = _k32.CreateFileW("nul", 0x40000000, 3, None, 3, 0, None)
    if _nul and _nul != -1:
        _k32.SetStdHandle(-12, _nul)
except Exception:
    pass
sys.stderr = open(os.devnull, "w")
if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except: pass

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

from config          import RTSP_URL, ENABLE_PTZ, SHOW_WINDOW
from detector        import Detector
from tracker         import Tracker
from ptz_controller  import PTZController


# ══════════════════════════════════════════════════════════════════════════════
#  Court region-of-interest
# ══════════════════════════════════════════════════════════════════════════════

def _load_court_polygon():
    """Load the manually-calibrated court polygon if present.

    Returns
    -------
    (np.ndarray of shape (N,1,2) int32, np.ndarray of shape (N,2) int32) or
    (None, None) when calibration is missing / disabled.
    """
    import json
    import numpy as np
    from config import COURT_FILE, COURT_FILTER_ENABLED
    if not COURT_FILTER_ENABLED:
        return None, None
    if not COURT_FILE.exists():
        log.info("No court calibration found at %s — court filter OFF. "
                 "Run calibrate_court.py to enable.", COURT_FILE)
        return None, None
    try:
        data = json.loads(COURT_FILE.read_text())
        pts  = np.array(data["polygon"], dtype=np.int32)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
            log.warning("court.json has invalid polygon shape; filter OFF")
            return None, None
        log.info("Court polygon loaded: %d points from %s",
                 len(pts), COURT_FILE.name)
        return pts.reshape(-1, 1, 2), pts
    except Exception as e:
        log.warning("Failed to read court.json (%s) — filter OFF", e)
        return None, None


def _filter_persons_by_court(persons, poly_cv, pad_px):
    """Keep only persons whose foot point lies inside the court polygon
    (with `pad_px` pixels of slack for sideline players)."""
    if poly_cv is None or not persons:
        return persons
    kept = []
    for p in persons:
        foot_x = (p[0] + p[2]) * 0.5
        foot_y = p[3]
        # cv2.pointPolygonTest with measureDist=True returns signed pixel
        # distance: positive inside, negative outside, 0 on the edge.
        d = cv2.pointPolygonTest(poly_cv, (float(foot_x), float(foot_y)), True)
        if d >= -pad_px:
            kept.append(p)
    return kept


# ══════════════════════════════════════════════════════════════════════════════
#  RTSP stream — tight-loop background thread, always stores the LATEST frame
#  (same pattern as the fast reference implementation)
# ══════════════════════════════════════════════════════════════════════════════

class RTSPStream:
    def __init__(self, url):
        self._url      = url
        self._frame    = None
        self._lock     = threading.Lock()
        self._running  = True
        self._cap_fps  = 0.0
        threading.Thread(target=self._loop, daemon=True, name="RTSP").start()

    # ── public ────────────────────────────────────────────────────────────────

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    @property
    def capture_fps(self):
        return self._cap_fps

    def stop(self):
        self._running = False

    # ── internals ─────────────────────────────────────────────────────────────

    def _open(self):
        log.info("Opening RTSP  %s", self._url)
        cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimal decode buffer
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            log.info("Stream opened: %dx%d @ %.1f fps (cap-reported)", w, h, fps)
        return cap

    def _loop(self):
        cnt, t0 = 0, time.time()
        first_frame_logged = False
        cap = self._open()

        while self._running:
            # Auto-reconnect when cap is broken
            if cap is None or not cap.isOpened():
                time.sleep(0.5)
                cap = self._open()
                first_frame_logged = False
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                log.warning("Stream read failed — reconnecting …")
                try: cap.release()
                except Exception: pass
                cap = None
                time.sleep(0.5)
                continue

            # Log actual pixel size on first decoded frame
            if not first_frame_logged:
                fh, fw = frame.shape[:2]
                log.info("First decoded frame: %dx%d  <-- actual camera output", fw, fh)
                first_frame_logged = True

            # Store latest frame (main thread picks it up at its own pace)
            with self._lock:
                self._frame = frame

            # Running FPS counter
            cnt += 1
            now = time.time()
            if now - t0 >= 1.0:
                self._cap_fps = cnt / (now - t0)
                cnt, t0 = 0, now

        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  Main loop
# ══════════════════════════════════════════════════════════════════════════════

_STATE_COLOUR = {
    "TRACKING":  (0, 255, 255),   # cyan
    "COASTING":  (0, 165, 255),   # orange
    "SEARCHING": (0, 80,  255),   # red
}


def main():
    stream  = RTSPStream(RTSP_URL)
    det     = Detector()
    tracker = Tracker()
    ptz     = PTZController()

    court_poly_cv, court_poly_xy = _load_court_polygon()
    from config import COURT_PADDING_PX, COURT_DRAW_OVERLAY

    if ENABLE_PTZ:
        log.info("Connecting ONVIF …")
        if ptz.connect():
            ptz.start()
            # If PTZ limits are loaded, send the camera to absolute (0,0)
            # so the dead-reckoned position matches the same origin used
            # during calibration. Without this, the saved limits won't
            # correspond to actual physical extremes after a restart.
            if ptz.limits_active():
                log.info("PTZ limits active — homing camera before tracking …")
                ptz.home_to_origin(wait_sec=3.0)
        else:
            log.warning("ONVIF offline — detection-only mode")

    fps_q = deque(maxlen=30)

    log.info("Waiting for first frame …")
    while stream.read() is None:
        time.sleep(0.05)
    log.info("Ready — press Q to quit.")

    while True:
        frame = stream.read()
        if frame is None:
            time.sleep(0.01)
            continue

        fps_q.append(time.monotonic())
        h, w = frame.shape[:2]

        # Detect
        t_det = time.perf_counter()
        persons = det.detect(frame)
        infer_ms = (time.perf_counter() - t_det) * 1000.0

        # Court ROI filter — drop persons whose feet are outside the court
        n_raw = len(persons)
        persons = _filter_persons_by_court(persons, court_poly_cv, COURT_PADDING_PX)
        n_off  = n_raw - len(persons)

        # Track
        pan_vel, tilt_vel, box, state, dbg = tracker.update(persons, w, h)

        # Command camera
        if ENABLE_PTZ and ptz.connected:
            if pan_vel is None:
                ptz.stop_move()
            else:
                ptz.move(pan_vel, tilt_vel)

        # Display
        if SHOW_WINDOW:
            vis = frame.copy()

            # Court polygon overlay (subtle cyan outline)
            if court_poly_xy is not None and COURT_DRAW_OVERLAY:
                cv2.polylines(vis, [court_poly_xy], isClosed=True,
                              color=(255, 200, 0), thickness=1, lineType=cv2.LINE_AA)

            # Draw persons — bright green = in-action, dim green = peripheral
            action_mask = dbg.get("action_mask", [])
            for i, (x1, y1, x2, y2) in enumerate(persons):
                in_act = action_mask[i] if i < len(action_mask) else True
                colour = (0, 230, 0) if in_act else (0, 100, 0)
                thickness = 2 if in_act else 1
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                              colour, thickness)

            # Action box — cyan outline (only in-action players + ball)
            if box:
                cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]),
                              (0, 255, 255), 2)

            # Lead-prediction dot — yellow circle where camera is aiming
            lead_px = dbg.get("lead_px")
            if lead_px:
                cv2.circle(vis, lead_px, 10, (0, 255, 255), -1)
                cv2.circle(vis, lead_px, 14, (0, 200, 200),  2)

            # Frame centre crosshair
            cv2.drawMarker(vis, (w // 2, h // 2), (255, 255, 255),
                           cv2.MARKER_CROSS, 20, 1)

            loop_fps = ((len(fps_q)-1) / (fps_q[-1]-fps_q[0]+1e-6)
                        if len(fps_q) > 1 else 0)
            col = _STATE_COLOUR[state]

            cv2.putText(vis,
                        f"CAP:{stream.capture_fps:.1f}  LOOP:{loop_fps:.1f}  INF:{infer_ms:.0f}ms  {w}x{h}",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200,200,200), 1, cv2.LINE_AA)
            cv2.putText(vis, state,
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.90, col, 2, cv2.LINE_AA)
            n_act  = dbg.get("n_action", 0)
            speed  = dbg.get("speed_norm", 0.0)
            court_tag = f"  OFF:{n_off}" if court_poly_cv is not None else ""
            cv2.putText(vis,
                        f"People:{len(persons)}  Action:{n_act}  Spd:{speed:.2f}{court_tag}",
                        (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200,200,200), 1, cv2.LINE_AA)
            if pan_vel is not None:
                cv2.putText(vis, f"pan={pan_vel:+.2f}  tilt={tilt_vel:+.2f}",
                            (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180,180,0), 1, cv2.LINE_AA)
            onvif_col = (0, 200, 0) if ptz.connected else (0, 60, 220)
            cv2.putText(vis, "ONVIF OK" if ptz.connected else "ONVIF OFFLINE",
                        (10, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.48, onvif_col, 1, cv2.LINE_AA)

            # PTZ limits status (only shown when calibration is loaded)
            if ptz.limits_active():
                pos = ptz.get_position()
                pan_hit, tilt_hit = ptz.last_clamp_hit()
                if pos is None:
                    lim_txt = "PTZ LIMITS: waiting GetStatus"
                    lim_col = (0, 60, 220)
                else:
                    p, t, _ = pos
                    lim_txt = f"PTZ LIMITS  pan={p:+.2f} tilt={t:+.2f}"
                    if pan_hit or tilt_hit:
                        lim_txt += "  CLAMP"
                        lim_col = (0, 165, 255)
                    else:
                        lim_col = (0, 200, 0)
                cv2.putText(vis, lim_txt, (10, 148),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, lim_col, 1, cv2.LINE_AA)

            cv2.imshow("PTZ Tracker", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    ptz.shutdown()
    stream.stop()
    cv2.destroyAllWindows()
    log.info("Stopped.")


if __name__ == "__main__":
    main()