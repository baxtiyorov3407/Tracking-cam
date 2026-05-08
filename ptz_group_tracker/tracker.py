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
from config import EMA_ALPHA, DEADBAND, START_BAND, SLOW_ZONE, MAX_VEL, PT_MIN_VEL, COAST_SEC, BALL_WEIGHT

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
        # ── No target ────────────────────────────────────────────────────────
        if not persons and ball is None:
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

        # ── Group bounding box (persons only) ─────────────────────────────────
        if persons:
            gx1 = min(b[0] for b in persons);  gy1 = min(b[1] for b in persons)
            gx2 = max(b[2] for b in persons);  gy2 = max(b[3] for b in persons)
            pw = (gx2 - gx1) * _PADDING;       ph = (gy2 - gy1) * _PADDING
            gx1 = max(0.0,            gx1 - pw);  gy1 = max(0.0,            gy1 - ph)
            gx2 = min(float(frame_w), gx2 + pw);  gy2 = min(float(frame_h), gy2 + ph)
            group_cx = (gx1 + gx2) * 0.5
            group_cy = (gy1 + gy2) * 0.5
            box = (int(gx1), int(gy1), int(gx2), int(gy2))
        else:
            # ball only — use ball centre as group centre
            group_cx = (ball[0] + ball[2]) * 0.5
            group_cy = (ball[1] + ball[3]) * 0.5
            box = (int(ball[0]), int(ball[1]), int(ball[2]), int(ball[3]))

        # ── Target centre: weighted blend towards ball when visible ───────────
        if ball is not None:
            ball_cx = (ball[0] + ball[2]) * 0.5
            ball_cy = (ball[1] + ball[3]) * 0.5
            cx = BALL_WEIGHT * ball_cx + (1.0 - BALL_WEIGHT) * group_cx
            cy = BALL_WEIGHT * ball_cy + (1.0 - BALL_WEIGHT) * group_cy
        else:
            cx = group_cx
            cy = group_cy

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

        if pv == 0.0 and tv == 0.0:
            return None, None, box, "TRACKING"
        return pv, tv, box, "TRACKING"