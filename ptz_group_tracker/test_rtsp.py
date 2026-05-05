"""
test_rtsp.py  —  Raw RTSP latency tester (no YOLO, no tracking).

Tests THREE different methods of reading the stream and shows which one is
lowest-latency. Watch the on-screen FPS and "frame age" counter.

Press Q to quit.
Press 1 / 2 / 3 to switch method live.

  Method 1 — cv2.VideoCapture plain read()           (baseline)
  Method 2 — background thread + shared frame        (our current approach)
  Method 3 — FFmpeg direct pipe (bypass OpenCV RTSP) (lowest latency possible)
"""

import cv2
import time
import threading
import subprocess
import numpy as np
from collections import deque

RTSP_URL = "rtsp://admin:123456@192.168.219.33:554/stream2"

# ── Common FFmpeg env flags (applied before cap.open) ─────────────────────────
import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|"
    "fflags;nobuffer|"
    "flags;low_delay|"
    "analyzeduration;0|"
    "probesize;32768|"
    "max_delay;0|"
    "allowed_media_types;video"
)

# Silence FFmpeg C-level stderr on Windows
import ctypes, sys
try:
    k32 = ctypes.windll.kernel32
    nul = k32.CreateFileW("nul", 0x40000000, 3, None, 3, 0, None)
    if nul and nul != -1:
        k32.SetStdHandle(-12, nul)
except Exception:
    pass
sys.stderr = open(os.devnull, "w")

# ══════════════════════════════════════════════════════════════════════════════
#  Method helpers
# ══════════════════════════════════════════════════════════════════════════════

def open_cap(url):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class BackgroundReader:
    """Continuously reads frames in a thread; main thread grabs latest."""
    def __init__(self, url):
        self._url    = url
        self._frame  = None
        self._lock   = threading.Lock()
        self._t      = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def read(self):
        with self._lock:
            return self._frame

    def _loop(self):
        cap = open_cap(self._url)
        while True:
            ret, f = cap.read()
            if not ret:
                cap.release()
                cap = open_cap(self._url)
                continue
            with self._lock:
                self._frame = f


class FFmpegPipeReader:
    """
    Decodes RTSP via a raw FFmpeg subprocess pipe — completely bypasses
    OpenCV's RTSP stack and its buffering. Lowest possible latency path.
    """
    def __init__(self, url):
        self._frame  = None
        self._w      = None
        self._h      = None
        self._lock   = threading.Lock()
        self._url    = url
        self._t      = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        # Wait until first frame to know the resolution
        for _ in range(100):
            if self._w is not None:
                break
            time.sleep(0.05)

    def read(self):
        with self._lock:
            return self._frame

    def _loop(self):
        # Probe dimensions first with a short ffprobe call
        try:
            out = subprocess.check_output([
                "ffprobe", "-v", "error",
                "-rtsp_transport", "tcp",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                self._url,
            ], timeout=10, stderr=subprocess.DEVNULL).decode().strip()
            w, h = map(int, out.split(","))
        except Exception:
            w, h = 1280, 720   # fallback

        with self._lock:
            self._w, self._h = w, h

        cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-analyzeduration", "0",
            "-probesize", "32768",
            "-i", self._url,
            "-vf", "fps=25",          # cap at 25 fps
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-an",                    # no audio
            "-f", "rawvideo",
            "pipe:1",
        ]
        frame_bytes = w * h * 3
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=frame_bytes * 2,
            )
            while True:
                raw = proc.stdout.read(frame_bytes)
                if len(raw) < frame_bytes:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
                with self._lock:
                    self._frame = frame.copy()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  Main test loop
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Initialising readers — please wait ~5 s …")

    bg   = BackgroundReader(RTSP_URL)
    pipe = FFmpegPipeReader(RTSP_URL)

    method  = 2          # start with background-thread method
    fps_dq  = deque(maxlen=30)
    prev_t  = time.monotonic()

    print("Window open. Press 1/2/3 to switch method, Q to quit.")
    print("  1 = plain cap.read()  2 = background thread  3 = FFmpeg pipe")

    # Method 1 cap (opened once, kept alive)
    cap1 = open_cap(RTSP_URL)

    while True:
        t0 = time.monotonic()

        if method == 1:
            ret, frame = cap1.read()
            if not ret:
                cap1.release()
                cap1 = open_cap(RTSP_URL)
                continue
            src_label = "Method 1: plain cap.read()"
        elif method == 2:
            frame = bg.read()
            if frame is None:
                time.sleep(0.01)
                continue
            src_label = "Method 2: background thread"
        else:
            frame = pipe.read()
            if frame is None:
                time.sleep(0.01)
                continue
            src_label = "Method 3: FFmpeg pipe (lowest latency)"

        # FPS
        fps_dq.append(t0)
        fps = (len(fps_dq) - 1) / (fps_dq[-1] - fps_dq[0] + 1e-6) if len(fps_dq) > 1 else 0

        # Render HUD
        vis = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR) \
              if frame.shape[1] > 1280 else frame.copy()
        h, w = vis.shape[:2]

        cv2.putText(vis, src_label,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, f"Display FPS: {fps:.1f}   Resolution: {frame.shape[1]}x{frame.shape[0]}",
                    (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(vis, "Press  1 / 2 / 3  to switch method    Q = quit",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 160, 160), 1, cv2.LINE_AA)

        # Blinking live indicator (flips every 0.5 s)
        if int(t0 * 2) % 2 == 0:
            cv2.circle(vis, (w - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(vis, "LIVE", (w - 70, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow("RTSP Latency Test", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("1"):
            method = 1
            print("Switched to Method 1")
        elif key == ord("2"):
            method = 2
            print("Switched to Method 2")
        elif key == ord("3"):
            method = 3
            print("Switched to Method 3")

    cap1.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
