"""
tracker.py  —  NBA-style basketball group tracker

Philosophy (mimicking a broadcast camera operator):
  1. Ball is the primary anchor — it is where the action is.
  2. Action-weighted centroid — players near the ball pull the camera hard;
     players in the far half-court barely register.
  3. Lead prediction — velocity of the action centre is tracked and the camera
     aims LEAD_TIME seconds *ahead*, so fast-breaks are anticipated rather
     than chased.
  4. Adaptive EMA — when action is fast the camera response tightens; during
     slow half-court possession it stays silky-smooth.
  5. Trapezoidal velocity profile — full speed when far off, linear ramp near
     centre, hard stop inside the dead-band.
  6. COASTING — when everyone disappears the motor holds direction for
     COAST_SEC seconds, then stops.

States:  TRACKING | COASTING | SEARCHING
"""

import math
import time

from config import (
    EMA_ALPHA, EMA_ALPHA_MAX, EMA_ALPHA_SCALE,
    DEADBAND, START_BAND, SLOW_ZONE, MAX_VEL, PT_MIN_VEL,
    COAST_SEC, BALL_WEIGHT,
    ACTION_SIGMA, LEAD_TIME, VEL_EMA_ALPHA, MIN_ACTION_WEIGHT,
    TILT_UP_BIAS, TILT_BIAS_THRESHOLD, CLOSE_MAX_VEL,
    PAN_PRIORITY_SCALE,
)

_PADDING = 0.06   # fractional padding around the action bounding box


# ── velocity profile ──────────────────────────────────────────────────────────

def _vel_profile(ema):
    """
    Map smoothed error [-1,+1] -> motor velocity [-1,+1].
    Full (capped) speed outside SLOW_ZONE, linear ramp between DEADBAND and
    SLOW_ZONE, hard stop inside DEADBAND.
    """
    e = abs(ema)
    if e < DEADBAND:
        return 0.0
    if e >= SLOW_ZONE:
        return math.copysign(MAX_VEL, ema)
    t = (e - DEADBAND) / (SLOW_ZONE - DEADBAND)
    return math.copysign(PT_MIN_VEL + t * (MAX_VEL - PT_MIN_VEL), ema)


# ── core algorithm helpers ────────────────────────────────────────────────────

def _player_weights(persons, ref_cx, ref_cy, sigma_px):
    """
    Compute an exponential proximity weight for each player relative to the
    reference point (ball centre, or person centroid when ball is absent).

    Returns list of (weight, px, py) in pixel space.
    """
    result = []
    for p in persons:
        px = (p[0] + p[2]) * 0.5
        py = (p[1] + p[3]) * 0.5
        dist = math.hypot(px - ref_cx, py - ref_cy)
        w = math.exp(-dist / max(sigma_px, 1.0))
        result.append((w, px, py))
    return result


def _weighted_centroid(weights):
    """
    Return (cx, cy) as the weight-normalised centroid of (w, px, py) triples.
    Falls back to a simple mean when total weight is negligible.
    """
    total = sum(w for w, _, _ in weights)
    if total < 1e-9:
        n = len(weights)
        return (sum(px for _, px, _ in weights) / n,
                sum(py for _, _, py in weights) / n)
    cx = sum(w * px for w, px, _ in weights) / total
    cy = sum(w * py for w, _, py in weights) / total
    return cx, cy


# ── Tracker ───────────────────────────────────────────────────────────────────

class Tracker:
    def __init__(self):
        # Pan / tilt EMA error accumulators
        self._pan_ema   = 0.0
        self._tilt_ema  = 0.0

        # Hysteresis flags (prevent chatter at dead-band edge)
        self._pan_on    = False
        self._tilt_on   = False

        # Coast timer
        self._coast_end = 0.0

        # Action-centre history for velocity estimation
        self._prev_cx   = None
        self._prev_cy   = None
        self._prev_t    = None

        # EMA-smoothed velocity of the action centre (pixels / second)
        self._vel_x     = 0.0
        self._vel_y     = 0.0

    # ── public API ────────────────────────────────────────────────────────────

    def update(self, persons, ball, frame_w, frame_h):
        """
        Parameters
        ----------
        persons  : list of [x1, y1, x2, y2]  (pixel coords)
        ball     : [x1, y1, x2, y2] or None
        frame_w  : int
        frame_h  : int

        Returns
        -------
        pan_vel    : float or None   -- motor command [-1, +1]
        tilt_vel   : float or None
        action_box : (x1,y1,x2,y2) or None  -- in-action player bounding box
        state      : "TRACKING" | "COASTING" | "SEARCHING"
        dbg        : dict with keys:
                       "action_mask"  - bool list (one per person)
                       "lead_px"      - (lx, ly) lead-prediction pixel or None
                       "speed_norm"   - normalised action speed (fraction/sec)
                       "n_action"     - number of in-action players
        """
        now = time.monotonic()

        # ── No targets at all ─────────────────────────────────────────────────
        if not persons and ball is None:
            self._prev_cx = self._prev_cy = self._prev_t = None
            self._vel_x  *= 0.85
            self._vel_y  *= 0.85

            _dbg_empty = {"action_mask": [], "lead_px": None,
                          "speed_norm": 0.0, "n_action": 0}

            if now < self._coast_end and (self._pan_on or self._tilt_on):
                pv = _vel_profile(self._pan_ema)  if self._pan_on  else 0.0
                tv = _vel_profile(self._tilt_ema) if self._tilt_on else 0.0
                if pv == 0.0 and tv == 0.0:
                    return None, None, None, "COASTING", _dbg_empty
                return pv, tv, None, "COASTING", _dbg_empty

            # Coast expired -- decay EMA and stop
            self._pan_ema  *= (1.0 - EMA_ALPHA)
            self._tilt_ema *= (1.0 - EMA_ALPHA)
            self._pan_on    = False
            self._tilt_on   = False
            return None, None, None, "SEARCHING", _dbg_empty

        # ── Reference anchor: ball centre or person centroid ──────────────────
        if ball is not None:
            ref_cx = (ball[0] + ball[2]) * 0.5
            ref_cy = (ball[1] + ball[3]) * 0.5
        elif persons:
            ref_cx = sum((p[0] + p[2]) * 0.5 for p in persons) / len(persons)
            ref_cy = sum((p[1] + p[3]) * 0.5 for p in persons) / len(persons)
        else:
            ref_cx, ref_cy = frame_w * 0.5, frame_h * 0.5

        # ── Weighted action centre ────────────────────────────────────────────
        # Players closer to the ball/ref get exponentially higher weight.
        # This naturally focuses the camera on the cluster near the action
        # rather than the geometric centre of all 10 players.
        if persons:
            sigma_px    = ACTION_SIGMA * frame_w
            wdata       = _player_weights(persons, ref_cx, ref_cy, sigma_px)
            group_cx, group_cy = _weighted_centroid(wdata)

            # Action mask: players with weight >= MIN_ACTION_WEIGHT * max_w
            max_w       = max(w for w, _, _ in wdata)
            threshold   = max_w * MIN_ACTION_WEIGHT
            action_mask = [w >= threshold for w, _, _ in wdata]
        else:
            wdata       = []
            action_mask = []
            group_cx, group_cy = ref_cx, ref_cy

        # ── Blend group centre toward ball ────────────────────────────────────
        if ball is not None:
            cx = BALL_WEIGHT * ref_cx + (1.0 - BALL_WEIGHT) * group_cx
            cy = BALL_WEIGHT * ref_cy + (1.0 - BALL_WEIGHT) * group_cy
        else:
            cx, cy = group_cx, group_cy

        # ── Velocity estimation ───────────────────────────────────────────────
        if self._prev_t is not None:
            dt = now - self._prev_t
            if 0.005 < dt < 1.0:
                raw_vx = (cx - self._prev_cx) / dt   # pixels / sec
                raw_vy = (cy - self._prev_cy) / dt
                self._vel_x += VEL_EMA_ALPHA * (raw_vx - self._vel_x)
                self._vel_y += VEL_EMA_ALPHA * (raw_vy - self._vel_y)
            else:
                # Stale gap -- bleed velocity to zero
                self._vel_x *= 0.6
                self._vel_y *= 0.6

        self._prev_cx, self._prev_cy, self._prev_t = cx, cy, now

        # ── Overflow detection (must happen before lead prediction) ──────────
        # overflow_norm: 0.0 = person fits in frame, 1.0 = fully overflowing.
        overflow_norm = 0.0
        if persons:
            avg_person_h     = sum(p[3] - p[1] for p in persons) / len(persons)
            person_size_frac = avg_person_h / frame_h
            overflow = max(0.0, person_size_frac - TILT_BIAS_THRESHOLD)
            overflow_norm = min(1.0, overflow / max(1e-6, 1.0 - TILT_BIAS_THRESHOLD))

        # ── Lead-prediction target ────────────────────────────────────────────
        # Both pan and tilt lead are suppressed when overflowing.
        # When a person is very close, a tiny physical movement = huge pixel
        # velocity, so predicting ahead causes overshoot in both axes.
        close_lead = LEAD_TIME * (1.0 - overflow_norm)
        lead_cx = max(0.0, min(float(frame_w), cx + close_lead * self._vel_x))
        lead_cy = max(0.0, min(float(frame_h), cy + close_lead * self._vel_y))

        # ── Normalised pan/tilt error relative to frame centre ────────────────
        pan_err  =  (lead_cx - frame_w * 0.5) / (frame_w * 0.5)
        tilt_err = -((lead_cy - frame_h * 0.5) / (frame_h * 0.5))

        # ── Tilt-up bias: only when person overflows the frame height ──────────
        if overflow_norm > 0.0:
            tilt_err += TILT_UP_BIAS * overflow_norm
            tilt_err  = max(-1.0, min(1.0, tilt_err))

        # ── Adaptive EMA: suppress speed-boost in overflow ────────────────────
        # In overflow the camera must move slowly and deliberately; a close
        # person moving fast in pixels must NOT make the camera react faster.
        speed_norm = math.hypot(self._vel_x / frame_w, self._vel_y / frame_h)
        alpha_boost = speed_norm * EMA_ALPHA_SCALE * (1.0 - overflow_norm)
        alpha = min(EMA_ALPHA_MAX, EMA_ALPHA + alpha_boost)

        # Pan gets full alpha — catch the person horizontally first.
        self._pan_ema += alpha * (pan_err - self._pan_ema)

        # ── Pan-priority tilt suppression ─────────────────────────────────────
        # When horizontal velocity is high, throttle tilt EMA so tilt corrections
        # wait until pan has caught up.  Once pan settles, tilt recovers fully.
        pan_speed_norm = abs(self._vel_x) / max(frame_w, 1)
        tilt_suppress  = 1.0 / (1.0 + PAN_PRIORITY_SCALE * pan_speed_norm)
        tilt_alpha     = alpha * tilt_suppress
        self._tilt_ema += tilt_alpha * (tilt_err - self._tilt_ema)

        # ── Hysteresis (start on START_BAND, stop on DEADBAND) ────────────────
        if self._pan_on:
            if abs(self._pan_ema)  < DEADBAND:    self._pan_on  = False
        else:
            if abs(self._pan_ema)  > START_BAND:  self._pan_on  = True
        if self._tilt_on:
            if abs(self._tilt_ema) < DEADBAND:    self._tilt_on = False
        else:
            if abs(self._tilt_ema) > START_BAND:  self._tilt_on = True

        # ── Motor velocity ────────────────────────────────────────────────────
        pv = _vel_profile(self._pan_ema)  if self._pan_on  else 0.0
        tv = _vel_profile(self._tilt_ema) if self._tilt_on else 0.0

        # In overflow: cap BOTH pan and tilt speed to CLOSE_MAX_VEL.
        # Blends smoothly: full cap at overflow_norm=1, no cap at 0.
        if overflow_norm > 0.0:
            close_cap = MAX_VEL + (CLOSE_MAX_VEL - MAX_VEL) * overflow_norm
            if pv != 0.0:
                pv = math.copysign(min(abs(pv), close_cap), pv)
            if tv != 0.0:
                tv = math.copysign(min(abs(tv), close_cap), tv)

        # ── Action bounding box (in-action players + ball only) ───────────────
        in_action = [p for p, m in zip(persons, action_mask) if m]
        if in_action:
            ax1 = min(p[0] for p in in_action)
            ay1 = min(p[1] for p in in_action)
            ax2 = max(p[2] for p in in_action)
            ay2 = max(p[3] for p in in_action)
            if ball is not None:
                ax1 = min(ax1, ball[0]); ay1 = min(ay1, ball[1])
                ax2 = max(ax2, ball[2]); ay2 = max(ay2, ball[3])
            pw = (ax2 - ax1) * _PADDING; ph = (ay2 - ay1) * _PADDING
            ax1 = max(0,       int(ax1 - pw)); ay1 = max(0,       int(ay1 - ph))
            ax2 = min(frame_w, int(ax2 + pw)); ay2 = min(frame_h, int(ay2 + ph))
            box = (ax1, ay1, ax2, ay2)
        elif ball is not None:
            box = (int(ball[0]), int(ball[1]), int(ball[2]), int(ball[3]))
        else:
            box = None

        # Refresh coast timer
        self._coast_end = now + COAST_SEC

        dbg = {
            "action_mask": action_mask,
            "lead_px":     (int(lead_cx), int(lead_cy)),
            "speed_norm":  speed_norm,
            "n_action":    sum(action_mask),
        }

        if pv == 0.0 and tv == 0.0:
            return None, None, box, "TRACKING", dbg
        return pv, tv, box, "TRACKING", dbg

