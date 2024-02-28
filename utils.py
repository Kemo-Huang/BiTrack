from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Calibration:
    def __init__(self, calib_file):
        with open(calib_file) as f:
            lines = f.readlines()
        obj = lines[2].strip().split(" ")[1:]
        self.P2 = np.array(obj, dtype=np.float32).reshape(3, 4)
        obj = lines[3].strip().split(" ")[1:]
        self.P3 = np.array(obj, dtype=np.float32).reshape(3, 4)
        obj = lines[4].strip().split(" ")[1:]
        self.R0 = np.array(obj, dtype=np.float32).reshape(3, 3)
        obj = lines[5].strip().split(" ")[1:]
        self.V2C = np.array(obj, dtype=np.float32).reshape(3, 4)

        # Camera intrinsics
        fu = self.P2[0, 0]
        fv = self.P2[1, 1]
        tx = self.P2[0, 3] / (-fu)
        ty = self.P2[1, 3] / (-fv)
        self.c = self.P2[:2, 2]
        self.f = np.array((fu, fv), dtype=np.float32)
        self.t = np.array((tx, ty), dtype=np.float32)

    def rect_to_lidar(self, pts_rect):
        return (pts_rect @ self.R0 - self.V2C[np.newaxis, :, 3]) @ self.V2C[:, :3]

    def lidar_to_rect(self, pts_lidar):
        return (pts_lidar @ self.V2C[:, :3].T + self.V2C[np.newaxis, :, 3]) @ self.R0.T

    def rect_to_img(self, pts_rect):
        pts_2d = pts_rect @ self.P2[:, :3].T + self.P2[np.newaxis, :, 3]
        a = pts_2d[:, 0:2]
        b = np.repeat(pts_rect[:, 2:3], 2, axis=1)
        pts_img = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        pts_rect_depth = pts_2d[:, 2] - self.P2[2, 3]
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2), pts_depth: (N)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth

    def img_to_rect(self, pts_img, depth_rect):
        """
        :param pts_img: (N, 2)
        :param depth_rect: (N)
        :return: (N, 3)
        """
        pts_xy = (pts_img - self.c) * np.repeat(
            depth_rect[:, np.newaxis], 2, axis=1
        ) / self.f + self.t
        return np.concatenate((pts_xy, depth_rect[:, np.newaxis]), axis=1)


class KittiObjectTemplate:
    def __init__(self):
        self.cls_type = ""
        self.truncation = -1
        self.occlusion = -1
        self.alpha = 0.0
        self.h = 0.0
        self.w = 0.0
        self.l = 0.0
        self.loc = np.zeros(3, dtype=np.float32)
        self.ry = 0.0
        self.tracking_score = 1.0
        self.box2d = np.zeros(4, dtype=int)
        self.img_hw = (375, 1242)

    def get_level(self):
        height = self.box2d[3] - self.box2d[1] + 1
        if height >= 40 and self.truncation <= 0.15 and self.occlusion <= 0:
            return 0  # Easy
        elif height >= 25 and self.truncation <= 0.3 and self.occlusion <= 1:
            return 1  # Moderate
        elif height >= 25 and self.truncation <= 0.5 and self.occlusion <= 2:
            return 2  # Hard
        else:
            return -1

    def from_lidar_box(
        self, lidar_box: np.ndarray, calib: Calibration, cls_type=None, score=None
    ):
        if cls_type is not None:
            self.cls_type = cls_type
        if score is not None:
            self.tracking_score = score

        l, w, h = lidar_box[3:6]
        self.h = h
        self.w = w
        self.l = l
        xyz = np.copy(lidar_box[:3])
        xyz[2] -= h / 2
        self.loc = calib.lidar_to_rect(xyz[np.newaxis])[0]
        self.ry = -lidar_box[6] - np.pi / 2
        self.alpha = -np.arctan2(-lidar_box[1], lidar_box[0]) + self.ry

        corners = boxes_to_corners_3d(lidar_box[np.newaxis])[0]
        pts_img, _ = calib.lidar_to_img(corners)
        min_uv = np.min(pts_img, axis=0)
        max_uv = np.max(pts_img, axis=0)
        self.box2d = np.array(
            [
                np.clip(min_uv[0], a_min=0, a_max=self.img_hw[1] - 1),
                np.clip(min_uv[1], a_min=0, a_max=self.img_hw[0] - 1),
                np.clip(max_uv[0], a_min=0, a_max=self.img_hw[1] - 1),
                np.clip(max_uv[1], a_min=0, a_max=self.img_hw[0] - 1),
            ],
            dtype=np.float32,
        )

        if (self.box2d[2] - self.box2d[0]) * (self.box2d[3] - self.box2d[1]) == 0:
            raise ValueError

        return self

    def to_lidar_box(self, calib: Calibration):
        loc_lidar = calib.rect_to_lidar(self.loc[np.newaxis])[0]
        loc_lidar[2] += self.h / 2
        angle = angle_in_range(-(np.pi / 2 + self.ry))
        lidar_box = np.concatenate(
            (loc_lidar, np.stack([self.l, self.w, self.h, angle]))
        )
        return lidar_box


class KittiObject3d(KittiObjectTemplate):
    def __init__(self, line: str = None, img_hw=None):
        if line is None:
            super().__init__()
        else:
            label = line.strip().split()
            self.cls_type = label[0]
            self.truncation = int(label[1])
            # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
            self.occlusion = int(label[2])
            self.alpha = float(label[3])
            self.box2d = np.array(
                (float(label[4]), float(label[5]), float(label[6]), float(label[7]))
            )
            self.h = float(label[8])
            self.w = float(label[9])
            self.l = float(label[10])
            self.loc = np.array(
                (float(label[11]), float(label[12]), float(label[13])), dtype=np.float32
            )
            self.ry = float(label[14])
            self.tracking_score = float(label[15]) if label.__len__() == 16 else -1.0
        if img_hw is not None:
            self.img_hw = img_hw

    def serialize(self):
        """Converts object to string with new line"""
        return (
            f"{self.cls_type} {self.truncation} {self.occlusion} {self.alpha} "
            f"{' '.join(self.box2d.astype(str).tolist())} {self.h} {self.w} {self.l} "
            f"{' '.join(self.loc.astype(str).tolist())} {self.ry} {self.tracking_score}\n"
        )


class KittiTrack3d(KittiObjectTemplate):
    def __init__(
        self,
        line: str = None,
        object3d: KittiObject3d = None,
        sample_id: int = None,
        tracking_id: int = None,
        img_hw=None,
    ):
        if line is None:
            if object3d is None:
                super().__init__()
            else:
                self.cls_type = object3d.cls_type
                self.truncation = object3d.truncation
                self.occlusion = object3d.occlusion
                self.alpha = object3d.alpha
                self.box2d = object3d.box2d
                self.h = object3d.h
                self.w = object3d.w
                self.l = object3d.l
                self.loc = object3d.loc
                self.ry = object3d.ry
                self.tracking_score = object3d.tracking_score
                self.img_hw = object3d.img_hw

            self.sample_id = sample_id
            self.tracking_id = tracking_id
            if img_hw is not None:
                self.img_hw = img_hw
        else:
            label = line.strip().split()
            self.sample_id = int(label[0])
            self.tracking_id = int(label[1])
            self.cls_type = label[2]
            self.truncation = int(label[3])
            # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
            self.occlusion = int(label[4])
            self.alpha = float(label[5])
            self.box2d = np.array(
                (float(label[6]), float(label[7]), float(label[8]), float(label[9]))
            )
            self.h = float(label[10])
            self.w = float(label[11])
            self.l = float(label[12])
            self.loc = np.array(
                (float(label[13]), float(label[14]), float(label[15])), dtype=np.float32
            )
            self.ry = float(label[16])
            self.tracking_score = float(label[17]) if label.__len__() == 18 else -1.0

    def to_obj(self):
        obj = KittiObject3d()
        obj.cls_type = self.cls_type
        obj.truncation = self.truncation
        # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        obj.occlusion = self.occlusion
        obj.alpha = self.alpha
        obj.box2d = self.box2d
        obj.h = self.h
        obj.w = self.w
        obj.l = self.l
        obj.loc = self.loc
        obj.ry = self.ry
        obj.tracking_score = self.tracking_score
        return obj

    def serialize(self):
        return (
            f"{self.sample_id} {self.tracking_id} {self.cls_type} {self.truncation} {self.occlusion} {self.alpha} "
            f"{' '.join(self.box2d.astype(str).tolist())} {self.h} {self.w} {self.l} "
            f"{' '.join(self.loc.astype(str).tolist())} {self.ry} {self.tracking_score}\n"
        )


def get_objects_from_label(label_file, track=False, as_dict=False):
    with open(label_file, "r") as f:
        lines = f.readlines()
    obj_cls = KittiTrack3d if track else KittiObject3d
    objects = [obj_cls(line) for line in lines]
    if as_dict:
        if track:
            frame_dict = {}
            for o in objects:
                frame = f"{o.sample_id:06}"
                if frame not in frame_dict:
                    frame_dict[frame] = [o]
                else:
                    frame_dict[frame].append(o)
            objects = {
                frame: {
                    "name": np.array([o.cls_type for o in v]),
                    "truncated": np.array([o.truncation for o in v]),
                    "occluded": np.array([o.occlusion for o in v]),
                    "alpha": np.array([o.alpha for o in v]),
                    "bbox": np.array([o.box2d for o in v]).reshape(-1, 4),
                    "dimensions": np.array([(o.l, o.h, o.w) for o in v]).reshape(-1, 3),
                    "location": np.array([o.loc for o in v]).reshape(-1, 3),
                    "rotation_y": np.array([o.ry for o in v]),
                    "score": np.array([o.tracking_score for o in v]),
                }
                for frame, v in frame_dict.items()
            }
        else:
            objects = {
                "name": np.array([o.cls_type for o in objects]),
                "truncated": np.array([o.truncation for o in objects]),
                "occluded": np.array([o.occlusion for o in objects]),
                "alpha": np.array([o.alpha for o in objects]),
                "bbox": np.array([o.box2d for o in objects]).reshape(-1, 4),
                "dimensions": np.array([(o.l, o.h, o.w) for o in objects]).reshape(
                    -1, 3
                ),
                "location": np.array([o.loc for o in objects]).reshape(-1, 3),
                "rotation_y": np.array([o.ry for o in objects]),
                "score": np.array([o.tracking_score for o in objects]),
            }
    return objects


def map_tracks_by_frames(
    seq_tracks: List[KittiTrack3d],
) -> Dict[int, List[KittiTrack3d]]:
    out = {}
    for track in seq_tracks:
        if track.sample_id in out:
            out[track.sample_id].append(track)
        else:
            out[track.sample_id] = [track]
    return out


def rotate_points_along_z(points: np.ndarray, angles: np.ndarray):
    """Batch-based points rotation

    Args:
        points (np.ndarray): (N, M, D)
        angles (np.ndarray): (N,)

    Returns:
        np.ndarray: rotated points (N, M, D)
    """
    cosa = np.cos(angles)
    sina = np.sin(angles)
    zeros = np.zeros(len(points))
    ones = np.ones(len(points))
    rot_matrix = np.stack(
        (cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones), axis=1
    ).reshape(-1, 3, 3)
    return np.concatenate(
        (np.matmul(points[:, :, 0:3], rot_matrix), points[:, :, 3:]), axis=-1
    )


def rotate_points_bev(points: np.ndarray, angles: np.ndarray):
    """Batch-based points rotation

    Args:
        points (np.ndarray): (N, M, D)
        angles (np.ndarray): (N,)

    Returns:
        np.ndarray: rotated points (N, M, D)
    """
    cosa = np.cos(angles)
    sina = np.sin(angles)
    rot_matrix = np.stack(
        (
            cosa,
            sina,
            -sina,
            cosa,
        ),
        axis=1,
    ).reshape(-1, 2, 2)
    return np.concatenate(
        (np.matmul(points[:, :, 0:2], rot_matrix), points[:, :, 2:]), axis=-1
    )


def boxes_to_corners_3d(boxes3d: np.ndarray):
    """
        4 -------- 5
       /|         /|
      7 -------- 6 .
      | |        | |
      . 0 -------- 1
      |/         |/
      3 -------- 2

    Args:
        boxes3d: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
        np.ndarry: (N, 8, 3)
    """
    template = (
        np.array(
            (
                [1, 1, -1],
                [1, -1, -1],
                [-1, -1, -1],
                [-1, 1, -1],
                [1, 1, 1],
                [1, -1, 1],
                [-1, -1, 1],
                [-1, 1, 1],
            )
        )
        / 2
    )  # (8, 3)
    corners3d = boxes3d[:, np.newaxis, 3:6] * template[np.newaxis]
    corners3d = rotate_points_along_z(corners3d, boxes3d[:, 6])
    corners3d += boxes3d[:, np.newaxis, 0:3]
    return corners3d


def get_boxes2d_from_boxes3d(lidar_boxes: np.ndarray, calib: Calibration, img_shape):
    n = lidar_boxes.shape[0]
    corners = boxes_to_corners_3d(lidar_boxes).reshape(n * 8, 3)
    pts_img, _ = calib.lidar_to_img(corners)
    pts_img = pts_img.reshape(n, 8, 2)
    min_uv = np.min(pts_img, axis=1)
    max_uv = np.max(pts_img, axis=1)
    boxes2d = np.stack(
        [
            np.clip(min_uv[:, 0], a_min=0, a_max=img_shape[1] - 1),
            np.clip(min_uv[:, 1], a_min=0, a_max=img_shape[0] - 1),
            np.clip(max_uv[:, 0], a_min=0, a_max=img_shape[1] - 1),
            np.clip(max_uv[:, 1], a_min=0, a_max=img_shape[0] - 1),
        ],
        axis=1,
    )
    return boxes2d


def boxes_to_corners_bev(boxes3d: np.ndarray) -> np.ndarray:
    """
    x: front, y: left
    0 --- 1
    |     |
    3 --- 2

    Args:
        boxes3d: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
        np.ndarry: (N, 4, 2)
    """
    template = np.array(([1, 1], [1, -1], [-1, -1], [-1, 1])) / 2
    corners = boxes3d[:, np.newaxis, 3:5] * template[np.newaxis]
    corners = rotate_points_bev(corners, boxes3d[:, 6])
    corners += boxes3d[:, np.newaxis, 0:2]
    return corners


def draw_boxes_bev(ax: plt.Axes, boxes, color, ids=None):
    if ids is not None:
        assert len(boxes) == len(ids)
        for tid, box in zip(ids, boxes):
            center = (box[0] + box[2]) / 2
            ax.text(center[0], center[1], str(tid), color=color)
    boxes = np.concatenate((boxes, boxes[:, 0:1]), axis=1)
    for box in boxes:
        ax.plot(box[:, 0], box[:, 1], color=color)


def points_inside_boxes(boxes: np.ndarray, points: np.ndarray):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    :param points: (M, 3)
    :return: masks: (N, M)
    """
    local_points = points[np.newaxis, :, :3] - boxes[:, np.newaxis, :3]
    local_points = rotate_points_along_z(local_points, -boxes[:, 6])
    masks = np.all(
        np.less_equal(np.abs(local_points) * 2, boxes[:, np.newaxis, 3:6]), axis=2
    )
    return masks


def get_poses_from_file(oxts_file):
    """
    Reads oxts file for tracking.
    - lat:     latitude of the oxts-unit (deg)
    - lon:     longitude of the oxts-unit (deg)
    - alt:     altitude of the oxts-unit (m)
    - roll:    roll angle (rad),  0 = level, positive = left side up (-pi..pi)
    - pitch:   pitch angle (rad), 0 = level, positive = front down (-pi/2..pi/2)
    - yaw:     heading (rad),     0 = east,  positive = counter clockwise (-pi..pi)

    Args:
        oxts_file (str): oxts txt file
    """
    with open(oxts_file) as f:
        lines = f.readlines()
    R0_inv = None
    t0 = None
    poses = []
    for line in lines:
        lat, lon, alt, rx, ry, rz = map(float, line.split()[:6])
        # Computes mercator scale from latitude
        scale = np.cos(lat * np.pi / 180.0)
        # Converts lat/lon coordinates to mercator coordinates using mercator scale
        er = 6378137
        # translation vector
        t = np.array(
            (
                scale * lon * np.pi * er / 180,
                scale * er * np.log(np.tan((90 + lat) * np.pi / 360)),
                alt,
            )
        )
        # rotation matrix (OXTS RT3000 user manual, page 71/92)
        Rx = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(rx), -np.sin(rx)],
                [0.0, np.sin(rx), np.cos(rx)],
            ]
        )  # base => nav  (level oxts => rotated oxts)
        Ry = np.array(
            [
                [np.cos(ry), 0.0, np.sin(ry)],
                [0.0, 1.0, 0.0],
                [-np.sin(ry), 0.0, np.cos(ry)],
            ]
        )  # base => nav  (level oxts => rotated oxts)
        Rz = np.array(
            [
                [np.cos(rz), -np.sin(rz), 0.0],
                [np.sin(rz), np.cos(rz), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )  # base => nav  (level oxts => rotated oxts)
        R = np.dot(Rz, np.dot(Ry, Rx))

        # normalize translation and rotation (start at 0/0/0)
        if R0_inv is None:
            R0_inv = R.T
            t0 = t
            R = np.eye(3)
            t = np.zeros(3)
        else:
            R = np.dot(R0_inv, R)
            t -= t0
        poses.append((R, t))
    return poses


def get_lidar_boxes_from_objs(objs: List[KittiObject3d], calib: Calibration):
    if len(objs) == 0:
        return np.empty((0, 7))
    dims = np.array([[obj.l, obj.h, obj.w] for obj in objs])
    loc = np.stack([obj.loc for obj in objs], axis=0)
    ry = np.array([obj.ry for obj in objs])

    loc_lidar = calib.rect_to_lidar(loc)
    l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
    loc_lidar[:, 2] += h[:, 0] / 2
    boxes_lidar = np.concatenate(
        [loc_lidar, l, w, h, -(np.pi / 2 + ry[:, np.newaxis])], axis=1
    )
    for i in range(len(objs)):
        boxes_lidar[i, 6] = angle_in_range(boxes_lidar[i, 6])
    return boxes_lidar


def get_global_boxes_from_lidar(
    boxes_lidar: np.ndarray, pose: Tuple[np.ndarray, np.ndarray]
):
    """Converts boxes from lidar coor. to global coor.

    Args:
        boxes_lidar (np.ndarray): (N, 7)
        pose: (R, t)
    """
    loc = boxes_lidar[:, :3]
    R, t = pose
    loc_global = np.dot(loc, R) + t
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.concatenate(
        (loc_global, boxes_lidar[:, 3:6], boxes_lidar[:, 6:7] + yaw), axis=1
    )


def get_lidar_boxes_from_global(
    boxes_global: np.ndarray, pose: Tuple[np.ndarray, np.ndarray]
):
    """Converts boxes from global coor. to lidar coor.

    Args:
        boxes_global (np.ndarray): (N, 7)
        pose: (R, t)
    """
    loc = boxes_global[:, :3]
    R, t = pose
    loc_lidar = np.dot(loc - t, R.T)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.concatenate(
        (loc_lidar, boxes_global[:, 3:6], boxes_global[:, 6:7] - yaw), axis=1
    )


def visualize_trajectories(
    trajectories: List[np.ndarray], other_boxes: np.ndarray = None
):
    fig, ax = plt.subplots()
    ax.axis("equal")
    for i, boxes in enumerate(trajectories):
        if isinstance(boxes, list):
            if len(boxes) > 0:
                boxes = np.stack(boxes)
            else:
                continue
        # elif isinstance(boxes, np.ndarray):
        #     boxes = boxes.reshape(-1, 7)
        x = -boxes[:, 1]
        y = boxes[:, 0]
        ax.quiver(
            x[:-1],
            y[:-1],
            x[1:] - x[:-1],
            y[1:] - y[:-1],
            units="xy",
            angles="xy",
            scale=1,
            color=f"C{i}",
            headaxislength=4,
        )
        ax.scatter(x, y, s=10, marker="o", facecolors="none", edgecolors=f"C{i}")
        # ax.plot(-boxes[:, 1], boxes[:, 0], '-o', markersize=3)
    if other_boxes is not None and len(other_boxes) > 0:
        ax.scatter(-other_boxes[:, 1], other_boxes[:, 0], s=2)
    plt.show()


def crop_points_from_boxes(
    points: np.ndarray, boxes: np.ndarray, front_only=True
) -> List[np.ndarray]:
    """
    Args:
        points (np.ndarray): (N, 3 + C)
        boxes (np.ndarray): (M, 7)
        front_only (bool, optional): _description_. Defaults to True.

    Returns:
        List[np.ndarray]: list of points, len = M
    """
    if front_only:
        points = points[points[:, 0] > 0]
    return [points[mask] for mask in points_inside_boxes(boxes, points[:, :3])]


def compute_iou_2d(
    bboxes1: np.ndarray, bboxes2: np.ndarray, do_ioa=False
) -> np.ndarray:
    """Computes 2D IoU. layout: (x0, y0, x1, y1)

    Args:
        bboxes1 (np.ndarray): (N, 4)
        bboxes2 (np.ndarray): (M, 4)
        do_ioa (bool, optional): Whether to compute IoA for bboxes1. Defaults to False.
    Returns:
        np.ndarray: (N, M)
    """
    min_ = np.minimum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
    max_ = np.maximum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
    intersection = np.maximum(min_[..., 2] - max_[..., 0], 0) * np.maximum(
        min_[..., 3] - max_[..., 1], 0
    )
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])

    if do_ioa:
        ioas = np.zeros_like(intersection)
        valid_mask = area1 > 0 + np.finfo("float").eps
        ioas[valid_mask, :] = (
            intersection[valid_mask, :] / area1[valid_mask][:, np.newaxis]
        )
        return ioas
    else:
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
            bboxes2[..., 3] - bboxes2[..., 1]
        )
        union = area1[:, np.newaxis] + area2[np.newaxis, :] - intersection
        intersection[area1 <= 0 + np.finfo("float").eps, :] = 0
        intersection[:, area2 <= 0 + np.finfo("float").eps] = 0
        intersection[union <= 0 + np.finfo("float").eps] = 0
        union[union <= 0 + np.finfo("float").eps] = 1
        ious = intersection / union
        return ious


def read_seqmap_file(seqmap_file):
    with open(seqmap_file) as f:
        lines = f.readlines()
    frame_num_dict = {}
    for line in lines:
        line = line.strip().split()
        frame_num_dict[line[0]] = int(line[-1])
    return frame_num_dict


def get_boxes2d_from_instance_map(instance_map: np.ndarray, ignored_inst_ids=None):
    inst_ids = np.unique(instance_map)[1:]
    if ignored_inst_ids is not None:
        inst_ids = np.isin(inst_ids, ignored_inst_ids, assume_unique=True, invert=True)
    boxes2d = []
    for inst_id in inst_ids:
        row_inds, col_inds = np.nonzero(instance_map == inst_id)
        boxes2d.append(
            np.array(
                (np.min(col_inds), np.min(row_inds), np.max(col_inds), np.max(row_inds))
            )
        )
    return np.stack(boxes2d) if len(boxes2d) > 0 else np.empty((0, 4))


def convert_trajectory_keys_from_track_id_to_frame(
    trajectories: Dict[int, Tuple[List[np.ndarray], List[KittiTrack3d]]]
):
    frame_dict = {}
    for trk_id, trajectories in trajectories.items():
        boxes, objs = trajectories
        for i in range(len(boxes)):
            frame = objs[i].sample_id
            assert trk_id == objs[i].tracking_id
            if frame not in frame_dict:
                frame_dict[frame] = [[boxes[i]], [objs[i]]]
            else:
                frame_dict[frame][0].append(boxes[i])
                frame_dict[frame][1].append(objs[i])
    return frame_dict


def sort_trajectory_by_frame(trajectory: Tuple[List[np.ndarray], List[KittiTrack3d]]):
    boxes, objs = trajectory
    trk_frames = np.array([obj.sample_id for obj in objs])
    inds = np.argsort(trk_frames)
    boxes = [boxes[idx] for idx in inds]
    objs = [objs[idx] for idx in inds]
    return boxes, objs


def write_kitti_trajectories_to_file(
    seq: str,
    trajectories: Dict[int, Tuple[List[np.ndarray], List[KittiTrack3d]]],
    txt_dir: Path,
):
    all_lines = []
    for _, objs in trajectories.values():
        all_lines += [obj.serialize() for obj in objs]
    with open(txt_dir / f"{seq}.txt", "w") as f:
        f.writelines(all_lines)


def read_kitti_trajectories_from_file(
    seq: str,
    txt_dir: Path,
    calib: Calibration,
    img_hw_dict: dict = None,
    raw_score=False,
):
    trajectories = {}
    tracks = get_objects_from_label(txt_dir / f"{seq}.txt", track=True)
    if raw_score:
        for trk in tracks:
            trk.tracking_score = sigmoid(trk.tracking_score)

    if img_hw_dict is not None:
        for track in tracks:
            track.img_hw = img_hw_dict[str(track.sample_id).zfill(6)]

    for track in tracks:
        if track.tracking_id not in trajectories:
            trajectories[track.tracking_id] = [[track.to_lidar_box(calib)], [track]]
        else:
            trajectories[track.tracking_id][0].append(track.to_lidar_box(calib))
            trajectories[track.tracking_id][1].append(track)

    return trajectories


def get_frustum_points(xyz, lidar2img, height=None, width=None, kitti=False):
    """
    Args:
        xyz: (N, 3)
        lidar2img: (3, 4)

    Returns:
        (M, 2) and mask (N)
    """
    if kitti:
        calib: Calibration = lidar2img
        pts_2d, pts_depth = calib.lidar_to_img(xyz)
    else:
        pts = np.hstack((xyz, np.ones((len(xyz), 1), dtype=np.float32)))
        pts = np.transpose(pts)  # (4, N)
        pts_2d = np.dot(lidar2img, pts)  # (3, N)
        pts_depth = np.transpose(pts_2d[2, :])
        pts_2d = np.transpose(pts_2d[:2, :] / pts_2d[2, :])  # (N, 2)

    pts_2d[:, (0, 1)] = pts_2d[:, (1, 0)]

    if height is not None and width is not None:
        mask1 = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < height)
        mask2 = (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < width)
        mask3 = pts_depth >= 0
        valid_proj_mask = mask1 & mask2 & mask3
    else:
        valid_proj_mask = pts_depth >= 0
    return pts_2d[valid_proj_mask], valid_proj_mask


def angle_in_range(angle: float):
    """Changes the angle to [0, 2pi)

    Args:
        angle (float):

    Returns:
        float:
    """
    while angle >= 2 * np.pi:
        angle -= np.pi * 2
    while angle < 0:
        angle += np.pi * 2
    return angle


def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.

    Input
        :param roll: The roll (rotation around x-axis) angle in radians.
        :param pitch: The pitch (rotation around y-axis) angle in radians.
        :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
        :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return [qx, qy, qz, qw]


class NuscenesObject:
    def __init__(self, data: dict = None) -> None:
        self.data = data
        if data:
            if "tracking_score" in data:
                self.tracking_score = data["tracking_score"]
            elif "detection_score" in data:
                self.tracking_score = data["detection_score"]
            else:
                self.tracking_score = 0
            self.tracking_id = None
            self.sample_id = data["sample_token"]
            self.loc = data["translation"]

    def serialize(self):
        self.data["tracking_id"] = str(self.tracking_id)
        if "detection_name" in self.data:
            self.data["tracking_name"] = self.data.pop("detection_name")
        return self.data

    def to_box(self):
        return np.array(
            self.data["translation"]
            + self.data["size"]
            + [quaternion_yaw(Quaternion(self.data["rotation"]))]
        )

    def from_box(
        self, box, sample_token, velocity, tracking_id, tracking_name, tracking_score
    ):
        self.tracking_id = tracking_id
        self.sample_id = sample_token
        self.tracking_score = tracking_score
        data = {
            "sample_token": sample_token,
            "translation": list(box[:3]),
            "size": list(box[3:6]),
            "rotation": get_quaternion_from_euler(0, 0, box[6]),
            "velocity": velocity,
            "tracking_id": tracking_id,
            "tracking_name": tracking_name,
            "tracking_score": tracking_score,
        }
        self.data = data
        return self
