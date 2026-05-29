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
    COAST_SEC,
    ACTION_SIGMA, LEAD_TIME, VEL_EMA_ALPHA, MIN_ACTION_WEIGHT,
    TILT_UP_BIAS, TILT_BIAS_THRESHOLD, CLOSE_MAX_VEL,
    PAN_PRIORITY_SCALE,
    GROUP_CLUSTER_DIST, GROUP_POWER,
    MOTION_WEIGHTING, MOTION_STATIC_FLOOR, MOTION_REF_SPEED,
    MOTION_MATCH_DIST, MOTION_TOTAL_FLOOR,
    MOTION_MOVER_THRESHOLD, MOTION_CONSENSUS_FRAC,
    MOTION_LEAD_TIME, MOTION_COHERENCE_MIN,
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


def _cluster_persons(persons, max_dist):
    """
    Single-linkage clustering of person boxes by centre-to-centre distance.
    Two persons closer than max_dist (pixels) belong to the same group.

    Returns a list of clusters; each cluster is a list of person indices
    into the original ``persons`` list.
    """
    n = len(persons)
    if n == 0:
        return []

    centres = [((p[0] + p[2]) * 0.5, (p[1] + p[3]) * 0.5) for p in persons]
    parent  = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    max_d2 = max_dist * max_dist
    for i in range(n):
        for j in range(i + 1, n):
            dx = centres[i][0] - centres[j][0]
            dy = centres[i][1] - centres[j][1]
            if dx * dx + dy * dy <= max_d2:
                union(i, j)

    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _blend_clusters(clusters, persons, power, motion_w=None):
    """
    Soft-blend all clusters to produce a single smooth target.

    Each cluster contributes its centroid with weight ``effective_size ** power``
    where ``effective_size`` is the sum of per-person motion weights (or the
    raw count when motion_w is None). This makes a moving group dominate over
    a static leftover person without ever "jumping" between clusters.

    Returns
    -------
    cx, cy : float
    cl_info : list of (idx_list, weight, ccx, ccy)
    max_w : float
    """
    cl_info = []
    total_w = 0.0
    cx_acc  = 0.0
    cy_acc  = 0.0
    max_w   = 0.0
    for idx_list in clusters:
        n   = len(idx_list)
        ccx = sum((persons[i][0] + persons[i][2]) * 0.5 for i in idx_list) / n
        ccy = sum((persons[i][1] + persons[i][3]) * 0.5 for i in idx_list) / n
        if motion_w is None:
            eff = float(n)
        else:
            eff = sum(motion_w[i] for i in idx_list)
        w   = max(eff, 1e-6) ** power
        cl_info.append((idx_list, w, ccx, ccy))
        total_w += w
        cx_acc  += w * ccx
        cy_acc  += w * ccy
        if w > max_w:
            max_w = w
    cx = cx_acc / total_w
    cy = cy_acc / total_w
    return cx, cy, cl_info, max_w


def _motion_weights(persons, prev_centres, dt, frame_w,
                    match_frac, ref_speed_frac, static_floor):
    """
    Estimate a per-person motion weight in [static_floor, 1.0] by matching
    each current detection to its nearest previous-frame centre.

    Returns (weights, centres, velocities) where velocities[i] is the
    estimated (vx, vy) in pixels/sec for person i (zero if no match).
    """
    centres = [((p[0] + p[2]) * 0.5, (p[1] + p[3]) * 0.5) for p in persons]
    if not prev_centres or dt <= 0.0 or dt > 1.0:
        return [1.0] * len(persons), centres, [(0.0, 0.0)] * len(persons)

    match_px      = match_frac * frame_w
    match_px_sq   = match_px * match_px
    ref_speed_px  = ref_speed_frac * frame_w
    weights, vels = [], []
    for cx, cy in centres:
        best_d2 = match_px_sq
        best    = None
        for px, py in prev_centres:
            d2 = (px - cx) ** 2 + (py - cy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best    = (px, py)
        if best is None:
            weights.append(1.0)
            vels.append((0.0, 0.0))
            continue
        vx = (cx - best[0]) / dt
        vy = (cy - best[1]) / dt
        speed_px = math.sqrt(best_d2) / dt
        m = min(1.0, speed_px / max(ref_speed_px, 1.0))
        weights.append(static_floor + (1.0 - static_floor) * m)
        vels.append((vx, vy))
    return weights, centres, vels


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

        # Previous-frame person centres (for per-person motion weighting)
        self._prev_centres = []
        self._prev_persons_t = None

    # ── public API ────────────────────────────────────────────────────────────

    def update(self, persons, frame_w, frame_h):
        """
        Parameters
        ----------
        persons  : list of [x1, y1, x2, y2]  (pixel coords)
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
        if not persons:
            self._prev_cx = self._prev_cy = self._prev_t = None
            self._prev_centres = []
            self._prev_persons_t = None
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

        # ── Per-person motion weighting ───────────────────────────────────────
        # Match each current detection to nearest previous-frame centre and
        # weight by speed: static people sink to MOTION_STATIC_FLOOR, movers
        # stay at ~1.0. This stops the camera from locking onto a lone
        # bystander after the action leaves the area.
        motion_w = None
        person_vels = [(0.0, 0.0)] * len(persons)
        total_motion_w = float(len(persons))
        if MOTION_WEIGHTING:
            dt_p = (now - self._prev_persons_t) if self._prev_persons_t else 0.0
            motion_w, new_centres, person_vels = _motion_weights(
                persons, self._prev_centres, dt_p, frame_w,
                MOTION_MATCH_DIST, MOTION_REF_SPEED, MOTION_STATIC_FLOOR,
            )
            self._prev_centres   = new_centres
            self._prev_persons_t = now
            total_motion_w = sum(motion_w)

        # ── Directional consensus (smooth blend, not a hard switch) ──────────
        # Compute a continuous strength in [0,1] from how many people are
        # moving and how aligned their velocities are. The strength then
        # smoothly:
        #   * reduces the weight of static people in the target blend,
        #   * scales an extra lead offset along the movers' mean velocity.
        # No hard threshold = no jerk when one person crosses the line.
        consensus_vx = consensus_vy = 0.0
        consensus_strength = 0.0
        if MOTION_WEIGHTING and motion_w is not None and len(persons) >= 2:
            movers = [i for i, w in enumerate(motion_w)
                      if w >= MOTION_MOVER_THRESHOLD]
            if len(movers) >= 2:
                moving_frac = len(movers) / len(persons)
                sum_vx = sum(person_vels[i][0] for i in movers)
                sum_vy = sum(person_vels[i][1] for i in movers)
                sum_sp = sum(math.hypot(*person_vels[i]) for i in movers)
                if sum_sp > 1e-6:
                    coherence = math.hypot(sum_vx, sum_vy) / sum_sp
                    # Ramp each factor from its MIN value to 1.0.
                    f_frac = max(0.0, min(1.0,
                        (moving_frac - MOTION_CONSENSUS_FRAC)
                        / max(1e-6, 1.0 - MOTION_CONSENSUS_FRAC)))
                    f_coh  = max(0.0, min(1.0,
                        (coherence - MOTION_COHERENCE_MIN)
                        / max(1e-6, 1.0 - MOTION_COHERENCE_MIN)))
                    consensus_strength = f_frac * f_coh
                    if consensus_strength > 0.0:
                        consensus_vx = sum_vx / len(movers)
                        consensus_vy = sum_vy / len(movers)
                        # Fade static people's weight smoothly toward zero
                        # as consensus grows; never a hard drop.
                        mover_set = set(movers)
                        motion_w = [
                            w if i in mover_set
                            else w * (1.0 - consensus_strength)
                            for i, w in enumerate(motion_w)
                        ]
                        total_motion_w = sum(motion_w)

        # If the scene only contains stationary leftovers, refuse to lock on
        # them — coast briefly, then SEARCH. This is what makes the camera
        # release a lone static person after a group leaves the frame.
        if MOTION_WEIGHTING and total_motion_w < MOTION_TOTAL_FLOOR:
            _dbg_static = {"action_mask": [False] * len(persons),
                           "lead_px": None, "speed_norm": 0.0, "n_action": 0}
            self._vel_x *= 0.85
            self._vel_y *= 0.85
            self._prev_cx = self._prev_cy = self._prev_t = None
            if now < self._coast_end and (self._pan_on or self._tilt_on):
                pv = _vel_profile(self._pan_ema)  if self._pan_on  else 0.0
                tv = _vel_profile(self._tilt_ema) if self._tilt_on else 0.0
                if pv == 0.0 and tv == 0.0:
                    return None, None, None, "COASTING", _dbg_static
                return pv, tv, None, "COASTING", _dbg_static
            self._pan_ema  *= (1.0 - EMA_ALPHA)
            self._tilt_ema *= (1.0 - EMA_ALPHA)
            self._pan_on    = False
            self._tilt_on   = False
            return None, None, None, "SEARCHING", _dbg_static

        # ── Cluster persons into groups, soft-blend by motion-weighted size ──
        # Each cluster's pull is (sum_of_motion_weights ** GROUP_POWER). A
        # moving group dominates a static one of the same head-count.
        cluster_dist_px = GROUP_CLUSTER_DIST * frame_w
        clusters        = _cluster_persons(persons, cluster_dist_px)
        blend_cx, blend_cy, cl_info, max_cl_w = _blend_clusters(
            clusters, persons, GROUP_POWER, motion_w=motion_w
        )

        # Refine with action sigma weighting: players nearer the blended
        # centre get a little extra pull (smooths motion within the group).
        sigma_px    = ACTION_SIGMA * frame_w
        wdata       = _player_weights(persons, blend_cx, blend_cy, sigma_px)
        group_cx, group_cy = _weighted_centroid(wdata)

        # action_mask: members of any cluster whose weight is at least
        # MIN_ACTION_WEIGHT of the largest cluster's weight. Isolated people
        # in tiny clusters are excluded from the on-screen action box.
        cluster_threshold = max_cl_w * MIN_ACTION_WEIGHT
        action_mask = [False] * len(persons)
        for idx_list, w, _ccx, _ccy in cl_info:
            if w >= cluster_threshold:
                for i in idx_list:
                    action_mask[i] = True

        # ── Action centre = blended group centroid ──────────────────────────
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
        # Two lead components, both suppressed when the subject overflows:
        #   * Baseline LEAD_TIME along the smoothed action-centre velocity
        #     — always on, acts as natural anticipation / look-space.
        #   * MOTION_LEAD_TIME scaled by consensus_strength along the
        #     movers' mean velocity — anticipates a coherent break early.
        base_lead = LEAD_TIME * (1.0 - overflow_norm)
        extra_lead = MOTION_LEAD_TIME * consensus_strength * (1.0 - overflow_norm)
        target_cx = cx + base_lead * self._vel_x + extra_lead * consensus_vx
        target_cy = cy + base_lead * self._vel_y + extra_lead * consensus_vy
        lead_cx = max(0.0, min(float(frame_w), target_cx))
        lead_cy = max(0.0, min(float(frame_h), target_cy))

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
        # Guard against a misconfigured ceiling (MAX < base) so the boost
        # remains meaningful.
        speed_norm   = math.hypot(self._vel_x / frame_w, self._vel_y / frame_h)
        alpha_boost  = speed_norm * EMA_ALPHA_SCALE * (1.0 - overflow_norm)
        alpha_ceil   = max(EMA_ALPHA, EMA_ALPHA_MAX)
        alpha        = min(alpha_ceil, EMA_ALPHA + alpha_boost)

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
            pw = (ax2 - ax1) * _PADDING; ph = (ay2 - ay1) * _PADDING
            ax1 = max(0,       int(ax1 - pw)); ay1 = max(0,       int(ay1 - ph))
            ax2 = min(frame_w, int(ax2 + pw)); ay2 = min(frame_h, int(ay2 + ph))
            box = (ax1, ay1, ax2, ay2)
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

