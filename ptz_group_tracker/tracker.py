"""
tracker.py  —  Group tracker with trapezoidal velocity profile.

Velocity profile:
  |error| > SLOW_ZONE              -> full speed  1.0
  DEADBAND < |error| <= SLOW_ZONE  -> linear ramp  PT_MIN_VEL .. 1.0
  |error| <= DEADBAND              -> stop  0.0

States:
  TRACKING  : target visible
  COASTING  : target gone, holding last velocity for COAST_SEC
  SEARCHING : coast expired, motor stopped
"""
import math
import time
from config import EMA_ALPHA, DEADBAND, START_BAND, SLOW_ZONE, MAX_VEL, PT_MIN_VEL, COAST_SEC

_PADDING = 0.08   # fractional padding added around the group bounding box


def _vel_profile(ema):
    """
    Map smoothed error [-1,+1] to motor velocity [-1,+1].
    Full (capped) speed outside SLOW_ZONE, linear ramp between SLOW_ZONE and DEADBAND.
    """
    e = abs(ema)
    if e < DEADBAND:
        return 0.0
    if e >= SLOW_ZONE:
        return math.copysign(MAX_VEL, ema)
    # linear ramp: DEADBAND -> PT_MIN_VEL,  SLOW_ZONE -> MAX_VEL
    t = (e - DEADBAND) / (SLOW_ZONE - DEADBAND)
    return math.copysign(PT_MIN_VEL + t * (MAX_VEL - PT_MIN_VEL), ema)


class Tracker:
    def __init__(self):
        self._pan_ema   = 0.0
        self._tilt_ema  = 0.0
        self._pan_on    = False   # hysteresis state (prevents chatter at deadband edge)
        self._tilt_on   = False
        self._coast_end = 0.0    # monotonic time when coast expires

    def update(self, persons, ball, frame_w, frame_h):
        """
        persons : list of [x1,y1,x2,y2]
        ball    : [x1,y1,x2,y2] or None

        Returns (pan_vel, tilt_vel, group_box, state)
          pan_vel / tilt_vel : float [-1,+1]  or  None = stop
          group_box          : (x1,y1,x2,y2) or None
          state              : "TRACKING" | "COASTING" | "SEARCHING"
        """
        boxes = list(persons)
        if ball is not None:
            boxes.append(ball)

        # ── No target ────────────────────────────────────────────────────────
        if not boxes:
            if time.monotonic() < self._coast_end and (self._pan_on or self._tilt_on):
                pv = _vel_profile(self._pan_ema)  if self._pan_on  else 0.0
                tv = _vel_profile(self._tilt_ema) if self._tilt_on else 0.0
                if pv == 0.0 and tv == 0.0:
                    return None, None, None, "COASTING"
                return pv, tv, None, "COASTING"
            # Coast expired — decay EMA and stop
            self._pan_ema  *= (1.0 - EMA_ALPHA)
            self._tilt_ema *= (1.0 - EMA_ALPHA)
            self._pan_on    = False
            self._tilt_on   = False
            return None, None, None, "SEARCHING"

        # ── Union bounding box of all detected objects ───────────────────────
        x1 = min(b[0] for b in boxes);  y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes);  y2 = max(b[3] for b in boxes)

        pw = (x2 - x1) * _PADDING;  ph = (y2 - y1) * _PADDING
        x1 = max(0.0,             x1 - pw);  y1 = max(0.0,             y1 - ph)
        x2 = min(float(frame_w),  x2 + pw);  y2 = min(float(frame_h),  y2 + ph)

        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5

        # Normalised error: -1 = left/down,  +1 = right/up
        pan_err  =  (cx - frame_w * 0.5) / (frame_w * 0.5)
        tilt_err = -((cy - frame_h * 0.5) / (frame_h * 0.5))

        # EMA smoothing (removes per-frame YOLO jitter)
        self._pan_ema  += EMA_ALPHA * (pan_err  - self._pan_ema)
        self._tilt_ema += EMA_ALPHA * (tilt_err - self._tilt_ema)

        # Hysteresis: start when error > START_BAND, stop when error < DEADBAND
        if self._pan_on:
            if abs(self._pan_ema)  < DEADBAND:    self._pan_on  = False
        else:
            if abs(self._pan_ema)  > START_BAND:  self._pan_on  = True
        if self._tilt_on:
            if abs(self._tilt_ema) < DEADBAND:    self._tilt_on = False
        else:
            if abs(self._tilt_ema) > START_BAND:  self._tilt_on = True

        # Velocity: trapezoidal profile (full speed far, ramp near centre)
        pv = _vel_profile(self._pan_ema)  if self._pan_on  else 0.0
        tv = _vel_profile(self._tilt_ema) if self._tilt_on else 0.0

        # Refresh coast timer while target is visible
        self._coast_end = time.monotonic() + COAST_SEC

        box = (int(x1), int(y1), int(x2), int(y2))
        if pv == 0.0 and tv == 0.0:
            return None, None, box, "TRACKING"
        return pv, tv, box, "TRACKING"