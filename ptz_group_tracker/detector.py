"""
detector.py  —  NanoDet-Plus ONNX person + ball detector.
Returns (persons, ball) where persons is a list of [x1,y1,x2,y2]
and ball is [x1,y1,x2,y2] or None.

Model auto-downloads on first run (~6 MB).
Typical CPU inference: 20-35 ms at 320x320.
"""
import math
import os
import urllib.request

import cv2
import numpy as np
import onnxruntime as ort

from config import MODEL_PATH, MODEL_URL, PERSON_CONF, BALL_CONF, NMS_IOU

_PERSON     = 0     # COCO class id
_BALL       = 32    # COCO class id (sports ball)
_INPUT_SIZE = 320
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

    @staticmethod
    def _softmax(x):
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=-1, keepdims=True)

    def detect(self, frame):
        """Returns (persons, ball)
        persons : list of [x1,y1,x2,y2]
        ball    : [x1,y1,x2,y2] or None
        """
        oh, ow = frame.shape[:2]

        # Pre-process
        inp = cv2.resize(frame, (_INPUT_SIZE, _INPUT_SIZE),
                         interpolation=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) - _MEAN) / _STD
        inp = inp.transpose(2, 0, 1)[None]          # (1, 3, H, W)

        # Inference
        out   = self._sess.run(None, {self._inp_name: inp})[0]
        preds = out[0] if out.ndim == 3 else out    # (anchors, 80 + 4*(reg_max+1))

        p_scores = preds[:, _PERSON]
        b_scores = preds[:, _BALL]
        mask     = (p_scores >= PERSON_CONF) | (b_scores >= BALL_CONF)
        idxs     = np.where(mask)[0]

        persons, ball, best_ball_conf = [], None, 0.0

        if idxs.size:
            labels = np.where(p_scores[idxs] >= b_scores[idxs], _PERSON, _BALL)
            confs  = np.where(labels == _PERSON, p_scores[idxs], b_scores[idxs])

            # class-specific threshold re-check
            valid = ((labels == _PERSON) & (confs >= PERSON_CONF)) | \
                    ((labels == _BALL)   & (confs >= BALL_CONF))
            idxs   = idxs[valid]
            labels = labels[valid]
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

                # Scale to original frame
                sx, sy = ow / _INPUT_SIZE, oh / _INPUT_SIZE
                bxs[:, [0, 2]] = (bxs[:, [0, 2]] * sx).clip(0, ow - 1)
                bxs[:, [1, 3]] = (bxs[:, [1, 3]] * sy).clip(0, oh - 1)

                for cls_id in (_PERSON, _BALL):
                    ci = np.where(labels == cls_id)[0]
                    if not ci.size:
                        continue
                    keep = _nms(bxs[ci], confs[ci])
                    for k in keep:
                        x1, y1, x2, y2 = (int(v) for v in bxs[ci[k]])
                        if x2 <= x1 or y2 <= y1:
                            continue
                        if cls_id == _PERSON:
                            persons.append([x1, y1, x2, y2])
                        else:
                            c = float(confs[ci[k]])
                            if c > best_ball_conf:
                                ball, best_ball_conf = [x1, y1, x2, y2], c

        return persons, ball