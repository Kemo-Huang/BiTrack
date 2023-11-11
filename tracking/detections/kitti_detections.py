from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from utils import (Calibration, KittiTrack3d, compute_iou_2d,
                   get_global_boxes_from_lidar, get_lidar_boxes_from_objs,
                   get_objects_from_label, sigmoid)

from .detections import Detections


def get_detection_data(
    det3d_file: Path,
    calib: Calibration,
    img_hw: tuple,
    lidar_dir: Path = None,
    seg_file: Path = None,
    det2d_file: Path = None,
    embed_dir: Path = None,
    pose: Tuple[np.ndarray, np.ndarray] = None,
    min_corr_pts=1,
    min_corr_iou=0.1,
    raw_score=False,
    score_thresh=0,
    recover_score_thresh=0.85,
) -> Tuple[Detections, Detections, Detections]:
    objs = get_objects_from_label(det3d_file, track=False)
    if len(objs) == 0:
        return None, None, None

    # Filters objects
    objs = [obj for obj in objs if obj.cls_type in ("Car", "Van")]
    if raw_score:
        objs = [obj for obj in objs if sigmoid(obj.tracking_score) >= score_thresh]
    else:
        objs = [obj for obj in objs if obj.tracking_score >= score_thresh]

    for obj in objs:
        obj.img_hw = img_hw

    # Converts to tracks
    objs = [
        KittiTrack3d(object3d=obj, sample_id=int(det3d_file.stem), tracking_id=None)
        for obj in objs
    ]

    # Converts boxes from rect to lidar
    boxes = get_lidar_boxes_from_objs(objs, calib)
    objs = np.array(objs)

    assert det2d_file is None or seg_file is None

    if det2d_file is not None:
        boxes2d = get_boxes2d_from_file(det2d_file)
        num_2d = len(boxes2d)
    elif seg_file is not None:
        instance_map = np.array(Image.open(seg_file))
        all_instance_ids = np.unique(instance_map)[1:]  # starts from 1
        num_2d = len(all_instance_ids)
    else:
        num_2d = -1

    if lidar_dir is None:
        if det2d_file is None and seg_file is None:
            # 3d box only
            good_detections = Detections(boxes, objs, None, None, None)
            bad_detections_3d = Detections(np.empty(0), np.empty(0), None, None, None)
            bad_detections_2d = None
        else:
            proj_boxes = np.array([obj.box2d for obj in objs]).reshape(-1, 4)
            if det2d_file is not None:
                # 3d box + 2d box
                corr_2d_inds, similarity = get_corr_2d_inds_hungarian(
                    calib,
                    img_hw,
                    proj_boxes,
                    "box",
                    boxes2d,
                    "box",
                    min_corr_pts,
                    min_corr_iou,
                )
            else:
                # 3d box + 2d mask
                corr_2d_inds, similarity = get_corr_2d_inds_hungarian(
                    calib,
                    img_hw,
                    proj_boxes,
                    "box",
                    instance_map,
                    "mask",
                    min_corr_pts,
                    min_corr_iou,
                    all_instance_ids,
                )
    else:
        inside_points = [
            np.fromfile(lidar_dir / f"{i}.bin", np.float32).reshape(-1, 4)[:, :3]
            for i in range(len(boxes))
        ]

        if det2d_file is None and seg_file is None:
            # 3d point only
            similarity = np.array([len(p) for p in inside_points])
            good_3d_mask = similarity >= min_corr_pts
            bad_3d_mask = ~good_3d_mask
            good_detections = Detections(
                boxes[good_3d_mask],
                objs[good_3d_mask],
                similarity[good_3d_mask],
                None,
                None,
            )
            bad_detections_3d = Detections(
                boxes[bad_3d_mask],
                objs[bad_3d_mask],
                similarity[bad_3d_mask],
                None,
                None,
            )
            bad_detections_2d = None

        else:
            if det2d_file is not None:
                # 3d point + 2d box
                corr_2d_inds, similarity = get_corr_2d_inds_hungarian(
                    calib,
                    img_hw,
                    inside_points,
                    "point",
                    boxes2d,
                    "box",
                    min_corr_pts,
                    min_corr_iou,
                )
            elif seg_file is not None:
                # 3d point + 2d mask
                corr_2d_inds, similarity = get_corr_2d_inds_hungarian(
                    calib,
                    img_hw,
                    inside_points,
                    "point",
                    instance_map,
                    "mask",
                    min_corr_pts,
                    min_corr_iou,
                    all_instance_ids,
                )

    if num_2d != -1:
        good_3d_mask = corr_2d_inds >= 0
        bad_3d_mask = ~good_3d_mask

        good_corr_2d_inds = corr_2d_inds[good_3d_mask]  # (num of good 3d boxes)
        bad_2d_inds = np.delete(np.arange(num_2d), good_corr_2d_inds)

        if embed_dir is None:
            good_embeds = None
            bad_embeds = None
        else:
            good_embeds = [
                np.load(embed_dir / f"{x}.npy").reshape(-1) for x in good_corr_2d_inds
            ]
            bad_embeds = [
                np.load(embed_dir / f"{x}.npy").reshape(-1) for x in bad_2d_inds
            ]
            if len(good_embeds) > 0:
                good_embeds = np.stack(good_embeds)
                good_embeds /= np.linalg.norm(good_embeds, axis=1, keepdims=True)
            if len(bad_embeds) > 0:
                bad_embeds = np.stack(bad_embeds)
                bad_embeds /= np.linalg.norm(bad_embeds, axis=1, keepdims=True)

        good_detections = Detections(
            boxes[good_3d_mask],
            objs[good_3d_mask],
            similarity[good_3d_mask],
            good_embeds,
            good_corr_2d_inds,
        )
        bad_detections_3d = Detections(
            boxes[bad_3d_mask], objs[bad_3d_mask], similarity[bad_3d_mask], None, None
        )
        bad_detections_2d = Detections(None, None, None, bad_embeds, bad_2d_inds)

    accept_mask = np.array(
        [obj.tracking_score >= recover_score_thresh for obj in bad_detections_3d.objs],
        dtype=bool,
    )
    accept_detections_3d = Detections(
        bad_detections_3d.boxes3d[accept_mask],
        bad_detections_3d.objs[accept_mask],
        None,
        None,
        None,
    )
    if len(accept_detections_3d) > 0:
        good_detections.append_3d(accept_detections_3d)
        bad_detections_3d = Detections(
            bad_detections_3d.boxes3d[~accept_mask],
            bad_detections_3d.objs[~accept_mask],
            None
            if bad_detections_3d.similarity is None
            else bad_detections_3d.similarity[~accept_mask],
            None,
            None,
        )

    # Converts boxes from lidar to global (relative to the first frame)
    if pose is not None:
        good_detections.boxes3d = get_global_boxes_from_lidar(
            good_detections.boxes3d, pose
        )
        if bad_detections_3d is not None:
            bad_detections_3d.boxes3d = get_global_boxes_from_lidar(
                bad_detections_3d.boxes3d, pose
            )

    return good_detections, bad_detections_3d, bad_detections_2d


def get_boxes2d_from_file(filename):
    """
    Ignores object types
    """
    with open(filename, "r") as f:
        lines = f.readlines()
    objs = [line.split()[:-1] for line in lines]
    return np.array(objs, dtype=np.float32).reshape(-1, 5)


def get_corr_2d_inds_hungarian(
    calib: Calibration,
    img_hw: tuple,
    data_3d: List[np.ndarray],
    type_3d: str,
    data_2d: np.ndarray,
    type_2d: str,
    min_corr_pts=1,
    min_corr_iou=0.1,
    all_inst_ids=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gets the corresponding 2D object indices of 3D objects via the Hungarian algorithm.

    Args:
        calib (Calibration): calibration
        img_hw (tuple): (H, W)
        data_3d (List[np.ndarray]): a list of object points or a list of 3D boxes
        type_3d (str): point or box
        data_2d (np.ndarray): instance image map (H, W) or 2D boxes (x1, y1, x2, y2)
        type_2d (str): mask or box
        min_corr_pts (int, optional): minimum number of points inside box. Defaults to 1.
        min_corr_iou (float, optional): minimum point iou, points inside instance over points inside box. Defaults to 0.
        all_inst_ids: all unique instance ids. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            An int array of 2d indices with the same length of the boxes, -1 if no corresponding 2d measurement.
            An int array of similarities with the same length of the boxes
    """
    N = len(data_3d)
    assert type_3d == "point" or type_3d == "box"
    assert type_2d == "mask" or type_2d == "box"
    if type_3d == "point" or type_2d == "mask":
        thresh = min_corr_pts
    else:
        thresh = min_corr_iou

    if type_2d == "mask":
        if all_inst_ids is None:
            all_inst_ids = np.unique(data_2d)[1:]
        M = len(all_inst_ids)
        inst_id2idx = {x: i for i, x in enumerate(all_inst_ids)}
    elif type_2d == "box":
        M = len(data_2d)

    final_2d_inds = np.ones(N, dtype=int) * -1
    final_similarity = np.zeros(N, dtype=int)

    if type_3d == "point":
        similarity_matrix = np.zeros((N, M), dtype=int)

        for i, points in enumerate(data_3d):  # for each 3d object
            if len(points) == 0:
                continue

            # projects points to image
            pts2d, depth = calib.lidar_to_img(points)
            valid_mask = (
                (depth > 0)
                & (pts2d[:, 0] >= 0)
                & (pts2d[:, 0] <= img_hw[1] - 1)
                & (pts2d[:, 1] >= 0)
                & (pts2d[:, 1] <= img_hw[0] - 1)
            )
            pts2d = pts2d[valid_mask]

            if type_2d == "mask":
                # computes the number of overlapped pixels for each 3d object and 2d instance
                pts2d = np.round(pts2d).astype(int)  # rounds to int
                cur_inst_ids = data_2d[pts2d[:, 1], pts2d[:, 0]]
                cur_inst_ids = cur_inst_ids[cur_inst_ids > 0]  # ignores background
                unique_ids, counts = np.unique(cur_inst_ids, return_counts=True)
                for j in range(len(unique_ids)):
                    similarity_matrix[i, inst_id2idx[unique_ids[j]]] = counts[j]

            elif type_2d == "box":
                # computes the number of inside pixels for each 3d object and 2d box
                ext_pts2d = pts2d[:, np.newaxis, :]
                valid_mask = (
                    (ext_pts2d[..., 0] >= data_2d[:, 0])
                    & (ext_pts2d[..., 0] <= data_2d[:, 2])
                    & (ext_pts2d[..., 1] >= data_2d[:, 1])
                    & (ext_pts2d[..., 1] <= data_2d[:, 3])
                )
                similarity_matrix[i] = np.sum(valid_mask, axis=0)  # (M,)

    elif type_3d == "box":
        if type_2d == "box":
            similarity_matrix = compute_iou_2d(data_3d, data_2d[:, :4])

        elif type_2d == "mask":
            similarity_matrix = np.zeros((N, M), dtype=int)
            all_mask_inds = [np.nonzero(data_2d == inst_id) for inst_id in all_inst_ids]
            for i, box in enumerate(data_3d):  # for each 3d object
                for j, mask_inds in enumerate(all_mask_inds):  # for each 2d instance
                    valid_mask = (
                        (mask_inds[0] >= box[0])
                        & (mask_inds[1] >= box[1])
                        & (mask_inds[0] <= box[2])
                        & (mask_inds[1] <= box[3])
                    )
                    similarity_matrix[i, j] = np.sum(valid_mask)

    row_inds, col_inds = linear_sum_assignment(similarity_matrix, maximize=True)
    valid_mask = similarity_matrix[row_inds, col_inds] >= thresh
    row_inds = row_inds[valid_mask]
    col_inds = col_inds[valid_mask]

    final_2d_inds[row_inds] = col_inds
    final_similarity[row_inds] = similarity_matrix[row_inds, col_inds]

    return final_2d_inds, final_similarity


def get_2d_box_inds_of_3d_boxes_hungarian(
    inside_points: List[np.ndarray],
    calib: Calibration,
    img_hw,
    boxes2d: np.ndarray,
    min_n_pts=1,
    min_pc_iou=0,
):
    N = len(inside_points)
    M = len(boxes2d)
    overlap_pc_cnts = np.zeros((N, M), dtype=int)
    total_pc_cnts = np.zeros(N, dtype=int)

    for i, points in enumerate(inside_points):  # for each 3d box
        if len(points) == 0:
            continue

        # projects points to image
        pts2d, depth = calib.lidar_to_img(points)
        valid_mask = (
            (depth > 0)
            & (pts2d[:, 0] >= 0)
            & (pts2d[:, 0] < img_hw[1])
            & (pts2d[:, 1] >= 0)
            & (pts2d[:, 1] < img_hw[0])
        )
        pts2d = pts2d[valid_mask]
        total_pc_cnts[i] = len(pts2d)

        # computes the number of inside pixels for each 3d box and 2d box
        ext_pts2d = pts2d[:, np.newaxis, :]
        valid_mask = (
            (ext_pts2d[..., 0] >= boxes2d[:, 1])
            & (ext_pts2d[..., 0] <= boxes2d[:, 3])
            & (ext_pts2d[..., 1] >= boxes2d[:, 0])
            & (ext_pts2d[..., 1] <= boxes2d[:, 2])
        )
        overlap_pc_cnts[i] = np.sum(valid_mask, axis=0)  # (M,)

    row_inds, col_inds = linear_sum_assignment(overlap_pc_cnts, maximize=True)
    valid_mask = overlap_pc_cnts[row_inds, col_inds] >= min_n_pts
    row_inds = row_inds[valid_mask]
    col_inds = col_inds[valid_mask]

    final_pc_counts = np.zeros(N, dtype=int)
    final_pc_counts[row_inds] = overlap_pc_cnts[row_inds, col_inds]
    final_box_ids = np.zeros(N, dtype=int)
    final_box_ids[row_inds] = col_inds

    pc_ious = np.divide(
        final_pc_counts,
        total_pc_cnts,
        out=np.zeros(N, dtype=float),
        where=total_pc_cnts != 0,
    )
    final_box_ids[pc_ious < min_pc_iou] = 0

    return final_box_ids, final_pc_counts
