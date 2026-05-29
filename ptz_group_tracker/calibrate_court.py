"""
calibrate_court.py  —  Interactive court calibration tool.

Run this ONCE per camera setup. It opens the RTSP stream, lets you freeze a
frame, then click the corners of the court (4 or more points, in order).
The resulting polygon is saved to court.json next to this file and used by
main.py to ignore detections outside the court (fans, passers-by) and to
keep the camera from panning off the playing area.

Controls
  SPACE  freeze / unfreeze the current frame
  Left-click   add a corner point (clockwise or counter-clockwise)
  U / Backspace  undo the last point
  R      restart (clear all points and unfreeze)
  ENTER  save polygon to court.json and exit  (needs >= 4 points)
  ESC / Q  quit without saving

IMPORTANT v1 limitation
  The polygon is stored in IMAGE pixel coordinates. It is only valid while
  the camera stays at the pan/tilt/zoom pose you calibrated at. The current
  tracker pans/tilts the camera to follow play, so calibrate at the "home"
  wide view that shows the entire court, then let main.py track from there.
  PTZ-aware (ray-based) calibration is a planned upgrade.
"""
import os, sys, json, time
from pathlib import Path

# Same FFmpeg / stderr setup as main.py so the stream behaves identically.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|stimeout;5000000|fflags;nobuffer|flags;low_delay|"
    "analyzeduration;0|probesize;32768|max_delay;0"
)
os.environ.setdefault("OPENCV_LOG_LEVEL",      "SILENT")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2
import numpy as np

from config import RTSP_URL, CAM_IP

OUT_FILE = Path(__file__).resolve().parent / "court.json"


def grab_frame(url, timeout_sec=10.0):
    """Open the RTSP stream and return the first good frame."""
    print(f"Opening RTSP {url} …")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        ret, frame = cap.read()
        if ret and frame is not None:
            cap.release()
            return frame
        time.sleep(0.1)
    cap.release()
    return None


class CalibratorState:
    def __init__(self):
        self.frozen_frame = None     # numpy frame once user presses SPACE
        self.points: list = []       # list of (x, y) in pixel coords

    def add_point(self, x, y):
        self.points.append((int(x), int(y)))

    def undo(self):
        if self.points:
            self.points.pop()

    def reset(self):
        self.frozen_frame = None
        self.points.clear()


def render(state, live_frame):
    """Build the display image — either the live feed or the frozen frame
    with overlay points/polygon and on-screen hints."""
    if state.frozen_frame is not None:
        img = state.frozen_frame.copy()
        banner = "FROZEN  -  Click court corners  |  U undo  R restart  ENTER save  ESC quit"
        banner_col = (0, 255, 255)
    else:
        img = live_frame.copy()
        banner = "LIVE  -  SPACE to freeze the current frame  |  ESC quit"
        banner_col = (200, 200, 200)

    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(img, banner, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, banner_col, 1, cv2.LINE_AA)

    # Draw clicked points + polygon
    for i, (x, y) in enumerate(state.points):
        cv2.circle(img, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(img, str(i + 1), (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    if len(state.points) >= 2:
        pts = np.array(state.points, dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=(len(state.points) >= 3),
                      color=(0, 255, 255), thickness=2)

    if state.frozen_frame is not None and len(state.points) >= 3:
        overlay = img.copy()
        cv2.fillPoly(overlay, [np.array(state.points, dtype=np.int32)],
                     (0, 200, 255))
        img = cv2.addWeighted(overlay, 0.15, img, 0.85, 0)

    cv2.putText(img, f"points: {len(state.points)}",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 0) if len(state.points) >= 4 else (0, 0, 255),
                1, cv2.LINE_AA)
    return img


def save(state):
    if len(state.points) < 4:
        print(f"Need at least 4 points; have {len(state.points)}. Not saving.")
        return False
    h, w = state.frozen_frame.shape[:2]
    data = {
        "polygon":   state.points,
        "frame_w":   int(w),
        "frame_h":   int(h),
        "camera_ip": CAM_IP,
        "saved_at":  time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": ("Polygon is in image-pixel coordinates of the camera at the "
                 "pose used during calibration. Valid only while the camera "
                 "stays near that pose."),
    }
    OUT_FILE.write_text(json.dumps(data, indent=2))
    print(f"Saved {len(state.points)} points to {OUT_FILE}")
    return True


def main():
    frame = grab_frame(RTSP_URL)
    if frame is None:
        print("ERROR: could not grab a frame from the camera. "
              "Check RTSP_URL in config.py and camera connectivity.")
        sys.exit(1)

    state = CalibratorState()
    win   = "Court Calibration"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1600, frame.shape[1]), min(900, frame.shape[0]))

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and state.frozen_frame is not None:
            state.add_point(x, y)

    cv2.setMouseCallback(win, on_mouse)

    # Keep a live-frame source for the "live" phase
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    live_frame = frame

    print("Calibration started. SPACE to freeze, then click court corners.")
    while True:
        if state.frozen_frame is None:
            ret, f = cap.read()
            if ret and f is not None:
                live_frame = f

        cv2.imshow(win, render(state, live_frame))
        key = cv2.waitKey(15) & 0xFF

        if key == 27 or key == ord('q'):           # ESC / Q
            print("Quit without saving.")
            break
        if key == 32:                              # SPACE
            if state.frozen_frame is None:
                state.frozen_frame = live_frame.copy()
                print("Frame frozen. Click court corners now.")
            else:
                state.reset()
                print("Unfrozen. Streaming live again.")
        if key in (ord('u'), 8):                   # U / Backspace
            state.undo()
        if key == ord('r'):                        # R
            state.reset()
            print("Reset.")
        if key == 13:                              # ENTER
            if save(state):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
