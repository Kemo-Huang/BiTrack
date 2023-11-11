import numpy as np
from filterpy.kalman import KalmanFilter

from utils import angle_in_range


class CVKalmanFilter:
    def __init__(self, box, p=10, q=2, r=1, ang_vel=True, vel_reinit=True):
        """Constant volocity (CV)-based Kalman filter.
        Args:
            box (np.ndarray): [x, y, z, dx, dy, dz, heading]
        """
        assert len(box) == 7
        box[6] = angle_in_range(box[6])

        self.ang_vel = ang_vel
        if ang_vel:
            # dim_x: [x, y, z, dx, dy, dz, heading, vx, vy, vz, vr]
            self.kf = KalmanFilter(dim_x=11, dim_z=7)
            self.kf.F = np.array(
                [  # state transition matrix
                    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ]
            )

        else:
            # dim_x: [x, y, z, dx, dy, dz, heading, vx, vy, vz]
            self.kf = KalmanFilter(dim_x=10, dim_z=7)
            self.kf.F = np.array(
                [  # state transition matrix
                    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ]
            )

        self.kf.x[:7] = box[:, np.newaxis]
        self.kf.H[:, :7] = np.eye(7)

        self.kf.P *= p  # initial state uncertainty
        self.kf.Q *= q  # process uncertainty
        self.kf.R *= r  # measurement uncertainty

        self.vel_initialized = False
        self.vel_reinit = vel_reinit

    def update(self, box):
        """
        Args:
            box (np.ndarray): [x, y, z, dx, dy, dz, heading]
        """
        box[6] = angle_in_range(box[6])
        if abs(box[6] - self.kf.x[6]) > np.pi:
            if box[6] > self.kf.x[6]:
                box[6] -= 2 * np.pi
            else:
                box[6] += 2 * np.pi
        if abs(box[6] - self.kf.x[6]) > np.pi / 2:
            if box[6] > self.kf.x[6]:
                box[6] -= np.pi
            else:
                box[6] += np.pi

        if self.vel_reinit and not self.vel_initialized:
            self.kf.x[7:10, 0] = box[:3] - self.kf.x[:3, 0]
            if self.ang_vel:
                self.kf.x[10, 0] = box[6] - self.kf.x[6, 0]
            self.kf.x[:7, 0] = box
            self.vel_initialized = True

        self.kf.update(box)
        self.kf.x[6] = angle_in_range(self.kf.x[6])

    def predict(self, t=1) -> np.array:
        """
        Advances the state vectors and returns the predicted bounding box estimate.
        """
        assert t > 0
        for _ in range(t):
            self.kf.predict()
        self.kf.x[6] = angle_in_range(self.kf.x[6])
        return self.kf.x[:7, 0]

    @property
    def x(self):
        return self.kf.x


class MAFilter:
    def __init__(self, box: np.ndarray):
        """Moving Average Filter

        Args:
            box (np.ndarray): [x, y, z, dx, dy, dz, heading]
        """
        assert len(box) == 7
        box[-1] = angle_in_range(box[-1])
        self.x = box
        self.v = np.zeros(7)
        self.step = 0

    def update(self, box):
        box[-1] = angle_in_range(box[-1])
        if self.step == 0:
            self.v = box - self.x
        else:
            self.v = (self.v + box - self.x) / 2
        self.v[-1] = angle_in_range(self.v[-1])
        self.x = box
        self.step += 1

    def predict(self, t=1):
        box = self.x + t * self.v
        box[-1] = angle_in_range(box[-1])
        return box
