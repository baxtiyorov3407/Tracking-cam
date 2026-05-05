"""
tracker.py - Group P-controller tracker with coast/momentum.

When the group is visible: normal P-control with EMA smoothing + hysteresis.
When the group disappears: camera keeps moving in the last direction for
COAST_DURATION seconds (e.g. player runs off the edge -> camera chases them).
"""

import time
import numpy as np
from config import (
    TARGET_FILL, GROUP_PADDING,
    KP_PAN, KP_TILT, EMA_ALPHA,
    DEADBAND, START_BAND, COAST_DURATION,
)


class GroupTracker:
    """
    Stateful group P-controller with edge-coast momentum.

    update(detections, frame_w, frame_h)
        -> (pan_vel, tilt_vel, group_box)

    pan_vel / tilt_vel : float [-1,+1] or None (stop).
    group_box          : (x1,y1,x2,y2) or None.
    """

    def __init__(self) -> None:
        self._pan_ema     = 0.0
        self._tilt_ema    = 0.0
        self._pan_moving  = False
        self._tilt_moving = False
        self._coast_until = 0.0   # monotonic time until coast is active

    def update(self, detections: dict, frame_w: int, frame_h: int) -> tuple:
        all_boxes = list(detections["persons"])
        if detections["ball"] is not None:
            all_boxes.append(detections["ball"])

        # ── No detections ─────────────────────────────────────────────────────
        if not all_boxes:
            now = time.monotonic()
            if now < self._coast_until:
                # Coast: keep last EMA frozen, return last velocity so the
                # camera continues moving in the direction the target left.
                pan_vel  = float(np.clip(KP_PAN  * self._pan_ema,  -1.0, 1.0)) if self._pan_moving  else 0.0
                tilt_vel = float(np.clip(KP_TILT * self._tilt_ema, -1.0, 1.0)) if self._tilt_moving else 0.0
                if pan_vel == 0.0 and tilt_vel == 0.0:
                    return None, None, None
                return pan_vel, tilt_vel, None
            # Coast expired — bleed EMA to zero and stop
            self._pan_ema  *= (1 - EMA_ALPHA)
            self._tilt_ema *= (1 - EMA_ALPHA)
            self._pan_moving  = False
            self._tilt_moving = False
            return None, None, None

        # ── Union bounding box ────────────────────────────────────────────────
        x1 = min(b[0] for b in all_boxes)
        y1 = min(b[1] for b in all_boxes)
        x2 = max(b[2] for b in all_boxes)
        y2 = max(b[3] for b in all_boxes)

        pw = (x2 - x1) * GROUP_PADDING
        ph = (y2 - y1) * GROUP_PADDING
        x1 = max(0.0,            x1 - pw)
        y1 = max(0.0,            y1 - ph)
        x2 = min(float(frame_w), x2 + pw)
        y2 = min(float(frame_h), y2 + ph)

        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5

        # Normalised error: -1 = far left/down, +1 = far right/up
        pan_err  =  (cx - frame_w * 0.5) / (frame_w * 0.5)
        tilt_err = -((cy - frame_h * 0.5) / (frame_h * 0.5))

        self._pan_ema  += EMA_ALPHA * (pan_err  - self._pan_ema)
        self._tilt_ema += EMA_ALPHA * (tilt_err - self._tilt_ema)

        # Hysteresis: start > START_BAND, stop < DEADBAND
        if self._pan_moving:
            if abs(self._pan_ema) < DEADBAND:    self._pan_moving  = False
        else:
            if abs(self._pan_ema) > START_BAND:  self._pan_moving  = True

        if self._tilt_moving:
            if abs(self._tilt_ema) < DEADBAND:   self._tilt_moving = False
        else:
            if abs(self._tilt_ema) > START_BAND: self._tilt_moving = True

        pan_vel  = float(np.clip(KP_PAN  * self._pan_ema,  -1.0, 1.0)) if self._pan_moving  else 0.0
        tilt_vel = float(np.clip(KP_TILT * self._tilt_ema, -1.0, 1.0)) if self._tilt_moving else 0.0

        # Refresh coast timer every frame we have a live detection
        self._coast_until = time.monotonic() + COAST_DURATION

        if pan_vel == 0.0 and tilt_vel == 0.0:
            return None, None, (int(x1), int(y1), int(x2), int(y2))

        return pan_vel, tilt_vel, (int(x1), int(y1), int(x2), int(y2))

    @property
    def pan_ema(self):  return self._pan_ema
    @property
    def tilt_ema(self): return self._tilt_ema