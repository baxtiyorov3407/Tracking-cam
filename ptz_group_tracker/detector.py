"""
detector.py  —  NanoDet-Plus ONNX person detector.
Returns a list of [x1,y1,x2,y2] person boxes.

Model auto-downloads on first run (~10 MB).
Typical CPU inference: 35-55 ms at 416x416.
"""
import math
import os
import urllib.request

import cv2
import numpy as np
import onnxruntime as ort

from config import (
    MODEL_PATH, MODEL_URL, PERSON_CONF, NMS_IOU,
    PERSON_MIN_HEIGHT, PERSON_MAX_ASPECT,
    CLAHE_ENABLED, CLAHE_CLIP, CLAHE_GRID,
    DETECTION_MIN_HITS, DETECTION_MAX_AGE,
)

_PERSON     = 0     # COCO class id
_INPUT_SIZE = 416
_MAX_DET    = 30

_MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
_STD  = np.array([ 57.375,  57.12,  58.395], dtype=np.float32).reshape(1, 1, 3)


# ── helpers ───────────────────────────────────────────────────────────────────

def _ensure_model():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 100 * 1024:
        return
    print(f"[DETECTOR] Downloading NanoDet-Plus ONNX (~6 MB) …")
    print(f"[DETECTOR]   -> {MODEL_PATH}")

    def _prog(n, bs, total):
        import sys
        if total > 0:
            sys.stdout.write(f"\r[DETECTOR] {min(100, n*bs*100//total):3d}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH), _prog)
    print()
    print("[DETECTOR] Download complete.")


def _letterbox(frame, size):
    """Resize frame into a square canvas with aspect-correct letterboxing.
    Returns (canvas, scale, pad_x, pad_y).
    Neutral fill = 114 (standard for COCO-trained models).
    """
    oh, ow = frame.shape[:2]
    scale  = size / max(ow, oh)
    nw     = int(round(ow * scale))
    nh     = int(round(oh * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas  = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x   = (size - nw) // 2
    pad_y   = (size - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


_IOU_MATCH = 0.25   # min IoU to link a raw detection to an existing stub


def _iou(a, b):
    """Intersection-over-Union of two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def _nms(boxes, scores):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size and len(keep) < _MAX_DET:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= NMS_IOU]
    return keep


# ── Detector class ────────────────────────────────────────────────────────────

class Detector:
    def __init__(self):
        _ensure_model()

        # Build anchor center priors once at init (constant for fixed input size)
        strides   = [8, 16, 32, 64]
        reg_max   = 7
        self._reg_max = reg_max
        self._proj    = np.arange(reg_max + 1, dtype=np.float32)

        centers, cstrides = [], []
        for s in strides:
            fh = math.ceil(_INPUT_SIZE / s)
            fw = math.ceil(_INPUT_SIZE / s)
            ys = np.arange(fh, dtype=np.float32) * s
            xs = np.arange(fw, dtype=np.float32) * s
            yy, xx = np.meshgrid(ys, xs, indexing="ij")
            pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
            centers.append(pts)
            cstrides.append(np.full(len(pts), s, dtype=np.float32))
        self._centers  = np.concatenate(centers)   # (N, 2)
        self._cstrides = np.concatenate(cstrides)  # (N,)

        # ONNX Runtime session
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        n = os.cpu_count() or 4
        so.intra_op_num_threads = max(1, min(4, n // 2))
        so.inter_op_num_threads = 1
        self._sess     = ort.InferenceSession(str(MODEL_PATH), sess_options=so,
                                              providers=["CPUExecutionProvider"])
        self._inp_name = self._sess.get_inputs()[0].name
        print(f"[DETECTOR] NanoDet-Plus ONNX ready  ({_INPUT_SIZE}x{_INPUT_SIZE})")

        # CLAHE for low-light contrast enhancement
        self._clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP,
            tileGridSize=(CLAHE_GRID, CLAHE_GRID)
        )

        # Temporal consistency stubs — each entry: {box, hits, age}
        self._stubs: list = []

    @staticmethod
    def _softmax(x):
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=-1, keepdims=True)

    def _update_person_stubs(self, raw_persons):
        """
        Temporal consistency filter.  raw_persons is the list of boxes detected
        in the current frame.  Returns only boxes that have appeared in at least
        DETECTION_MIN_HITS consecutive frames.
        """
        matched_stub = [False] * len(self._stubs)
        matched_det  = [False] * len(raw_persons)

        # Greedy nearest-stub matching by IoU
        for di, det in enumerate(raw_persons):
            best_iou, best_si = 0.0, -1
            for si, stub in enumerate(self._stubs):
                if matched_stub[si]:
                    continue
                iou = _iou(det, stub['box'])
                if iou > best_iou:
                    best_iou, best_si = iou, si
            if best_iou >= _IOU_MATCH and best_si >= 0:
                self._stubs[best_si]['box']   = det
                self._stubs[best_si]['hits'] += 1
                self._stubs[best_si]['age']   = 0
                matched_stub[best_si] = True
                matched_det[di]       = True

        # Age unmatched existing stubs; drop expired ones
        next_stubs = []
        for si, stub in enumerate(self._stubs):
            if not matched_stub[si]:
                stub['age'] += 1
            if stub['age'] <= DETECTION_MAX_AGE:
                next_stubs.append(stub)
        self._stubs = next_stubs

        # New stubs for unmatched raw detections (starts at hits=1)
        for di, det in enumerate(raw_persons):
            if not matched_det[di]:
                self._stubs.append({'box': det, 'hits': 1, 'age': 0})

        # Return boxes that have been confirmed enough times
        return [s['box'] for s in self._stubs if s['hits'] >= DETECTION_MIN_HITS]

    def detect(self, frame):
        """Returns persons: list of [x1,y1,x2,y2]"""
        oh, ow = frame.shape[:2]

        # ── Optional CLAHE contrast enhancement (helps in dark/evening frames) ─
        if CLAHE_ENABLED:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b_ch = cv2.split(lab)
            l = self._clahe.apply(l)
            frame = cv2.cvtColor(cv2.merge([l, a, b_ch]), cv2.COLOR_LAB2BGR)

        # Pre-process — letterbox to preserve aspect ratio (ball stays round)
        lbox, lb_scale, lb_pad_x, lb_pad_y = _letterbox(frame, _INPUT_SIZE)
        inp = (lbox.astype(np.float32) - _MEAN) / _STD
        inp = inp.transpose(2, 0, 1)[None]          # (1, 3, H, W)

        # Inference
        out   = self._sess.run(None, {self._inp_name: inp})[0]
        preds = out[0] if out.ndim == 3 else out    # (anchors, 80 + 4*(reg_max+1))

        p_scores = preds[:, _PERSON]
        mask     = p_scores >= PERSON_CONF
        idxs     = np.where(mask)[0]

        persons = []

        if idxs.size:
            confs  = p_scores[idxs]

            # threshold re-check
            valid  = confs >= PERSON_CONF
            idxs   = idxs[valid]
            confs  = confs[valid]

            if idxs.size:
                # Decode boxes via distribution integral
                reg  = preds[idxs, 80:].reshape(-1, 4, self._reg_max + 1)
                prob = self._softmax(reg)
                dist = (prob * self._proj).sum(axis=-1)      # (M, 4)
                dist = dist * self._cstrides[idxs, None]

                ctr  = self._centers[idxs]
                bxs  = np.zeros((len(idxs), 4), np.float32)
                bxs[:, 0] = ctr[:, 0] - dist[:, 0]
                bxs[:, 1] = ctr[:, 1] - dist[:, 1]
                bxs[:, 2] = ctr[:, 0] + dist[:, 2]
                bxs[:, 3] = ctr[:, 1] + dist[:, 3]

                # Undo letterbox → original frame coordinates
                bxs[:, [0, 2]] = ((bxs[:, [0, 2]] - lb_pad_x) / lb_scale).clip(0, ow - 1)
                bxs[:, [1, 3]] = ((bxs[:, [1, 3]] - lb_pad_y) / lb_scale).clip(0, oh - 1)

                keep = _nms(bxs, confs)
                for k in keep:
                    x1, y1, x2, y2 = (int(v) for v in bxs[k])
                    if x2 <= x1 or y2 <= y1:
                        continue
                    bh = y2 - y1
                    bw = x2 - x1
                    if bh < PERSON_MIN_HEIGHT:
                        continue
                    if bw / max(bh, 1) > PERSON_MAX_ASPECT:
                        continue
                    persons.append([x1, y1, x2, y2])

        # ── Temporal consistency filter ───────────────────────────────────
        persons = self._update_person_stubs(persons)

        return persons