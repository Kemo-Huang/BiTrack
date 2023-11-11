import numpy as np

from .motion_filters import CVKalmanFilter


class Track:
    global_cur_id = 1

    def __init__(
        self,
        box: np.ndarray,
        obj,
        offline=True,
        embed=None,
        momentum=0.9,
        p=10,
        q=2,
        ang_vel=True,
        vel_reinit=True,
    ):
        self.id = Track.global_cur_id
        Track.global_cur_id += 1
        self.filter = CVKalmanFilter(
            box, p=p, q=q, ang_vel=ang_vel, vel_reinit=vel_reinit
        )
        self.embed = embed
        self.misses = 0
        self.hits = 0
        self.offline = offline
        self.obj = obj
        self.new = True
        self.momentum = momentum
        if self.offline:
            self.max_hits = 0
            self.boxes = [box]
            self.objs = [obj]

    def update(self, box, obj, embed=None):
        self.filter.update(box)
        self.obj = obj
        if self.offline:
            self.boxes.append(box)
            self.objs.append(obj)

        if embed is not None:
            self.embed = self.momentum * self.embed + (1 - self.momentum) * embed
            self.embed /= np.linalg.norm(self.embed)

    def predict(self, t=1):
        return self.filter.predict(t)
