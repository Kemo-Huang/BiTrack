from typing import Dict, List, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from utils import KittiTrack3d, NuscenesObject


def box_size_weighted_mean(
    trajectories: Dict[int, Tuple[List[np.ndarray], List[KittiTrack3d]]],
    calib,
    img_hw_dict,
    exponent=45,
):
    for trk_id, trajectory in trajectories.items():
        boxes, objs = trajectory
        if isinstance(boxes, list):
            boxes = np.array(boxes)
        scores = np.array([x.tracking_score for x in objs])
        scores = scores**exponent
        boxes[:, 3:6] = np.sum(
            np.tile(scores[:, np.newaxis], (1, 3)) * boxes[:, 3:6], axis=0
        ) / np.sum(scores)

        new_objs = []
        for box, obj in zip(boxes, objs):
            new_objs.append(
                KittiTrack3d(
                    sample_id=obj.sample_id,
                    tracking_id=obj.tracking_id,
                    img_hw=img_hw_dict[str(obj.sample_id).zfill(6)],
                ).from_lidar_box(box, calib, obj.cls_type, obj.tracking_score)
            )
        trajectories[trk_id] = (boxes, new_objs)
    return trajectories


def box_size_weighted_mean_nusc(
    trajectories: Dict[int, Tuple[List[np.ndarray], List[NuscenesObject]]],
    exponent=45,
):
    for trk_id, trajectory in trajectories.items():
        boxes, objs = trajectory
        if isinstance(boxes, list):
            boxes = np.array(boxes)
        scores = np.array([x.tracking_score for x in objs])
        scores = scores**exponent
        boxes[:, 3:6] = np.sum(
            np.tile(scores[:, np.newaxis], (1, 3)) * boxes[:, 3:6], axis=0
        ) / np.sum(scores)

        new_objs = []
        for box, obj in zip(boxes, objs):
            new_objs.append(
                NuscenesObject(None).from_box(
                    box,
                    obj.data["sample_token"],
                    obj.data["velocity"],
                    obj.tracking_id,
                    obj.data["tracking_name"],
                    obj.tracking_score,
                )
            )
        trajectories[trk_id] = (boxes, new_objs)
    return trajectories


def gaussian_smoothing(
    trajectories: Dict[int, Tuple[List[np.ndarray], List[KittiTrack3d]]],
    calib,
    img_hw_dict,
    tau,
):
    for trk_id, trajectory in trajectories.items():
        boxes, objs = trajectory
        if isinstance(boxes, list):
            boxes = np.array(boxes)
        len_scale = np.clip(tau * np.log(tau**3 / len(objs)), tau**-1, tau**2)
        gpr = GaussianProcessRegressor(RBF(len_scale, "fixed"))
        t = np.array([obj.sample_id for obj in objs])[:, np.newaxis]
        boxes[:, 0] = gpr.fit(t, boxes[:, 0:1]).predict(t)
        boxes[:, 1] = gpr.fit(t, boxes[:, 1:2]).predict(t)
        boxes[:, 2] = gpr.fit(t, boxes[:, 2:3]).predict(t)
        # boxes[:, 6] = gpr.fit(t, boxes[:, 6:7]).predict(t)

        new_objs = []
        for box, obj in zip(boxes, objs):
            new_objs.append(
                KittiTrack3d(
                    sample_id=obj.sample_id,
                    tracking_id=obj.tracking_id,
                    img_hw=img_hw_dict[str(obj.sample_id).zfill(6)],
                ).from_lidar_box(box, calib, obj.cls_type, obj.tracking_score)
            )
        trajectories[trk_id] = (boxes, new_objs)

    return trajectories


def gaussian_smoothing_nusc(
    trajectories: Dict[int, Tuple[List[np.ndarray], List[NuscenesObject]]],
    tau,
):
    for trk_id, trajectory in trajectories.items():
        boxes, objs = trajectory
        if isinstance(boxes, list):
            boxes = np.array(boxes)
        len_scale = np.clip(tau * np.log(tau**3 / len(objs)), tau**-1, tau**2)
        gpr = GaussianProcessRegressor(RBF(len_scale, "fixed"))
        t = np.array([obj.sample_id for obj in objs])[:, np.newaxis]
        boxes[:, 0] = gpr.fit(t, boxes[:, 0:1]).predict(t)
        boxes[:, 1] = gpr.fit(t, boxes[:, 1:2]).predict(t)
        boxes[:, 2] = gpr.fit(t, boxes[:, 2:3]).predict(t)
        # boxes[:, 6] = gpr.fit(t, boxes[:, 6:7]).predict(t)

        new_objs = []
        for box, obj in zip(boxes, objs):
            new_objs.append(
                NuscenesObject().from_box(box, obj.sample_id, obj.data["velocity"], obj.tracking_id, obj.data["tracking_name"], obj.tracking_score)
            )
        trajectories[trk_id] = (boxes, new_objs)

    return trajectories