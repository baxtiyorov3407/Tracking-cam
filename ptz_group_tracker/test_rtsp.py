"""
test_rtsp.py  -  Minimum-latency RTSP live viewer for stream2.

Press T to toggle TCP / UDP transport.
Press Q to quit.

Tips if still delayed:
  1. On the camera web UI: Video -> Encoding -> reduce I-Frame Interval to 25
  2. Try UDP (press T) - skips TCP retransmission delay on local network
"""

import cv2, os, sys, ctypes, threading, time, numpy as np
from collections import deque

RTSP_URL = "rtsp://admin:123456@192.168.219.33:554/stream2"

# Silence FFmpeg C-level stderr on Windows
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
try:
    _k32 = ctypes.windll.kernel32
    _nul = _k32.CreateFileW("nul", 0x40000000, 3, None, 3, 0, None)
    if _nul and _nul != -1:
        _k32.SetStdHandle(-12, _nul)
except Exception:
    pass
sys.stderr = open(os.devnull, "w")


def _make_cap(url: str, transport: str) -> cv2.VideoCapture:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        f"rtsp_transport;{transport}|"
        "fflags;nobuffer+discardcorrupt|"
        "flags;low_delay|"
        "analyzeduration;0|"
        "probesize;32768|"
        "max_delay;0|"
        "reorder_queue_size;0|"
        "allowed_media_types;video"
    )
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class LiveReader:
    """
    Background thread that reads the RTSP stream as fast as possible.
    Main thread always gets the LATEST decoded frame.

    Key idea: between each real read() the thread calls grab() in a tight loop
    to flush any frames that accumulated while the OS was busy elsewhere.
    grab() does NOT decode - it is nearly free.  When grab() blocks > 15 ms we
    have caught up to the live network edge, so we retrieve() and decode that
    one frame.
    """

    def __init__(self, url: str, transport: str = "tcp"):
        self._url        = url
        self._transport  = transport
        self._frame      = None
        self._lock       = threading.Lock()
        self._rebuild    = False
        self._running    = True
        self._dropped    = 0
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def read(self):
        with self._lock:
            d = self._dropped
            self._dropped = 0
            return self._frame, d

    def switch(self, transport: str):
        self._transport = transport
        self._rebuild   = True

    def stop(self):
        self._running = False

    def _loop(self):
        cap = _make_cap(self._url, self._transport)
        while self._running:
            if self._rebuild:
                self._rebuild = False
                cap.release()
                time.sleep(0.3)
                cap = _make_cap(self._url, self._transport)
                continue

            # --- decode one frame ------------------------------------------
            ret, frame = cap.read()
            if not ret:
                cap.release()
                time.sleep(1.0)
                cap = _make_cap(self._url, self._transport)
                continue

            # --- drain stale buffered frames --------------------------------
            # grab() is fast (no decode).  It returns instantly when frames
            # are already buffered, and blocks ~1/fps when we are at the
            # live edge.  We discard buffered frames and decode only the
            # freshest one.
            dropped = 0
            while True:
                t0 = time.monotonic()
                ok = cap.grab()
                if not ok:
                    break
                dt = time.monotonic() - t0
                if dt > 0.015:        # blocked >15ms => live edge reached
                    r2, f2 = cap.retrieve()
                    if r2:
                        frame  = f2
                        dropped += 1
                    break
                dropped += 1          # grabbed a stale frame, keep draining

            with self._lock:
                self._frame    = frame
                self._dropped += dropped


def main():
    transport = "tcp"
    reader    = LiveReader(RTSP_URL, transport)
    fps_dq    = deque(maxlen=40)
    drop_dq   = deque(maxlen=40)

    print("Waiting for first frame …")
    for _ in range(200):
        f, _ = reader.read()
        if f is not None:
            break
        time.sleep(0.05)
    print("Stream open.  Press T = toggle TCP/UDP    Q = quit")

    while True:
        t0    = time.monotonic()
        frame, dropped = reader.read()
        if frame is None:
            time.sleep(0.01)
            continue

        fps_dq.append(t0)
        drop_dq.append(dropped)
        fps      = (len(fps_dq) - 1) / (fps_dq[-1] - fps_dq[0] + 1e-6) if len(fps_dq) > 1 else 0
        avg_drop = sum(drop_dq) / max(1, len(drop_dq))

        # resize for display
        h0, w0 = frame.shape[:2]
        if w0 > 1280:
            vis = cv2.resize(frame, (1280, int(h0 * 1280 / w0)), cv2.INTER_LINEAR)
        else:
            vis = frame.copy()
        dh, dw = vis.shape[:2]

        trans_col = (0, 200, 255) if transport == "udp" else (0, 255, 0)
        cv2.putText(vis, f"stream2  transport={transport.upper()}  {w0}x{h0}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.70, trans_col, 2, cv2.LINE_AA)
        cv2.putText(vis, f"FPS: {fps:.1f}    stale frames dropped/cycle: {avg_drop:.1f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        latency_hint = "low-latency" if avg_drop > 0.5 else "WARNING: no frames being dropped - buffer may be empty or slow"
        cv2.putText(vis, latency_hint,
                    (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (0, 255, 100) if avg_drop > 0.5 else (0, 100, 255), 1, cv2.LINE_AA)

        cv2.putText(vis, "T = toggle TCP/UDP    Q = quit",
                    (10, dh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 150, 150), 1, cv2.LINE_AA)

        # blinking live dot
        if int(t0 * 2) % 2 == 0:
            cv2.circle(vis, (dw - 28, 28), 10, (0, 0, 255), -1)
            cv2.putText(vis, "LIVE", (dw - 70, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow("Live RTSP Viewer", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("t"), ord("T")):
            transport = "udp" if transport == "tcp" else "tcp"
            reader.switch(transport)
            fps_dq.clear()
            drop_dq.clear()
            print(f"Switched to {transport.upper()}")

    reader.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()