"""
detector.py – YOLO26 people (and optional sports-ball) detector.

Returns a dict:
  'persons' : list of [x1, y1, x2, y2]
  'ball'    : [x1, y1, x2, y2] of the highest-confidence ball, or None
"""

from ultralytics import YOLO
from config import YOLO_MODEL, CONF_THRESH

_PERSON = 0    # COCO class: person
_BALL   = 32   # COCO class: sports ball (basketball, football, etc.)


class PeopleDetector:
    def __init__(self) -> None:
        self._model   = YOLO(YOLO_MODEL)
        self._classes = [_PERSON, _BALL]   # always track ball (sports court mode)

    def detect(self, frame) -> dict:
        """
        Run inference on a BGR frame.

        Returns
        -------
        dict with:
          'persons' : list[[x1,y1,x2,y2]]
          'ball'    : [x1,y1,x2,y2] or None
        """
        results = self._model(
            frame,
            verbose=False,
            conf=CONF_THRESH,
            imgsz=320,
            classes=self._classes,
        )[0]

        persons   = []
        ball      = None
        ball_conf = 0.0

        for box in results.boxes:
            cls  = int(box.cls[0])
            xyxy = box.xyxy[0].tolist()
            if cls == _PERSON:
                persons.append(xyxy)
            elif cls == _BALL:
                c = float(box.conf[0])
                if c > ball_conf:
                    ball      = xyxy
                    ball_conf = c

        return {"persons": persons, "ball": ball}

