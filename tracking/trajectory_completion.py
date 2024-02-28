from typing import Dict, List, Tuple

import numpy as np

from utils import (Calibration, KittiTrack3d, NuscenesObject,
                   sort_trajectory_by_frame, visualize_trajectories)

from .motion_filters import CVKalmanFilter
from .tracker import compute_ncd_matrix


def kf_predict_tail_boxes(
    frames: List[int], boxes: List[np.ndarray], tail_length
) -> List[np.ndarray]:
    assert len(frames) == len(boxes)
    n = len(frames)
    kf = CVKalmanFilter(boxes[0])
    for i in range(1, n):
        kf.predict(abs(frames[i] - frames[i - 1]))
        kf.update(boxes[i])
    return [kf.predict(1) for _ in range(tail_length)]


def linear_interpolation(
    trajectories: Dict[int, Tuple[List[np.ndarray], List[KittiTrack3d]]],
    calib: Calibration,
    img_hw_dict: dict,
    interp_max_interval: int,
    ignore_thresh=0.3,
    score_thresh=0,
    nms_thresh=0,
    visualize=False,
):
    candidates_dict = {}  # (trk_id, box, pred_len)
    existing_boxes = {}  # {frame: boxes}
    interp_nums = {}
    if visualize:
        vis_ids = []
    for trk_id, trajectory in trajectories.items():
        boxes, objs = trajectory
        interp_nums[trk_id] = len(boxes)  # before interp

        if len(objs) == 1:
            frame = objs[0].sample_id
            if frame not in existing_boxes:
                existing_boxes[frame] = [boxes[0]]
            else:
                existing_boxes[frame].append(boxes[0])
            continue

        # inserts middle boxes
        prev_obj = objs[0]
        prev_frame = prev_obj.sample_id
        prev_score = prev_obj.tracking_score
        prev_box = boxes[0]
        for i in range(1, len(objs)):
            cur_obj = objs[i]
            cur_box = boxes[i]
            cur_frame = cur_obj.sample_id
            cur_score = cur_obj.tracking_score

            if cur_frame not in existing_boxes:
                existing_boxes[cur_frame] = [cur_box]
            else:
                existing_boxes[cur_frame].append(cur_box)

            interval = cur_frame - prev_frame
            if interval > 1 and interval <= interp_max_interval:
                if visualize:
                    vis_ids.append(trk_id)
                box_diff = cur_box - prev_box
                score_diff = cur_score - prev_score
                for j in range(1, interval):
                    interp_frame = prev_frame + j
                    interp_box = prev_box + (box_diff * j / interval)  # interpolation
                    interp_score = prev_score + (score_diff * j / interval)
                    if interp_frame not in candidates_dict:
                        candidates_dict[interp_frame] = [
                            (trk_id, interp_box, interp_score)
                        ]
                    else:
                        candidates_dict[interp_frame].append(
                            (trk_id, interp_box, interp_score)
                        )
            prev_frame = cur_frame
            prev_box = cur_box

    if visualize:
        visualize_trajectories([trajectories[tid][0] for tid in vis_ids])
    for frame, candidates in candidates_dict.items():
        cand_ids = np.array([x[0] for x in candidates])
        cand_boxes = np.array([x[1] for x in candidates], dtype=np.float32).reshape(
            -1, 7
        )
        cand_scores = np.array([x[2] for x in candidates])

        if ignore_thresh > 0:
            # deletes the candidates which can be matched with the existing boxes in the same frame
            if frame in existing_boxes:
                ex_boxes = np.array(existing_boxes[frame], dtype=np.float32).reshape(
                    -1, 7
                )
                dis_aff_matrix = compute_ncd_matrix(cand_boxes, ex_boxes)
                valid_mask = np.max(dis_aff_matrix, axis=1) < ignore_thresh
                cand_ids = cand_ids[valid_mask]
                cand_boxes = cand_boxes[valid_mask]
                cand_scores = cand_scores[valid_mask]
            else:
                ex_boxes = np.empty((0, 7), dtype=np.float32)

        if score_thresh != 0:
            valid_mask = cand_scores >= score_thresh
            cand_scores = cand_scores[valid_mask]
            cand_ids = cand_ids[valid_mask]
            cand_boxes = cand_boxes[valid_mask]

        # NMS
        if nms_thresh != 0:
            order = np.argsort(cand_scores)
            nms_ids = []
            nms_boxes = []
            nms_scores = []
            while len(order) > 0:
                picked_box = cand_boxes[order[-1]]

                nms_ids.append(cand_ids[order[-1]])
                nms_boxes.append(picked_box)
                nms_scores.append(cand_scores[order[-1]])

                order = order[:-1]
                dis_aff_matrix = compute_ncd_matrix(
                    picked_box[np.newaxis], cand_boxes[order]
                )
                order = order[dis_aff_matrix[0] < nms_thresh]
            cand_ids = np.array(nms_ids)
            cand_boxes = np.array(nms_boxes)
            cand_scores = np.array(nms_scores)

        for cand_id, cand_box, cand_score in zip(cand_ids, cand_boxes, cand_scores):
            boxes, objs = trajectories[cand_id]
            if isinstance(boxes, list):
                boxes.append(cand_box)
            else:
                boxes = np.concatenate((boxes, cand_box[np.newaxis]))
            objs.append(
                KittiTrack3d(
                    sample_id=frame,
                    tracking_id=cand_id,
                    img_hw=img_hw_dict[str(frame).zfill(6)],
                ).from_lidar_box(
                    lidar_box=cand_box, calib=calib, cls_type="Car", score=cand_score
                )
            )
            trajectories[cand_id] = sort_trajectory_by_frame((boxes, objs))

    if visualize:
        visualize_trajectories([trajectories[tid][0] for tid in vis_ids])

    return trajectories


def linear_interpolation_nusc(
    trajectories: Dict[int, Tuple[List[np.ndarray], List[NuscenesObject]]],
    interp_max_interval: int,
    ignore_thresh=0.3,
    score_thresh=0,
    nms_thresh=0,
    visualize=False,
):
    candidates_dict = {}  # (trk_id, box, pred_len)
    existing_boxes = {}  # {frame: boxes}
    interp_nums = {}
    if visualize:
        vis_ids = []
    for trk_id, trajectory in trajectories.items():
        boxes, objs = trajectory
        interp_nums[trk_id] = len(boxes)  # before interp

        if len(objs) == 1:
            frame = objs[0].sample_id
            if frame not in existing_boxes:
                existing_boxes[frame] = [boxes[0]]
            else:
                existing_boxes[frame].append(boxes[0])
            continue

        # inserts middle boxes
        prev_obj = objs[0]
        prev_frame = prev_obj.sample_id
        prev_score = prev_obj.tracking_score
        prev_box = boxes[0]
        for i in range(1, len(objs)):
            cur_obj = objs[i]
            cur_box = boxes[i]
            cur_frame = cur_obj.sample_id
            cur_score = cur_obj.tracking_score

            if cur_frame not in existing_boxes:
                existing_boxes[cur_frame] = [cur_box]
            else:
                existing_boxes[cur_frame].append(cur_box)

            interval = cur_frame - prev_frame
            if interval > 1 and interval <= interp_max_interval:
                if visualize:
                    vis_ids.append(trk_id)
                box_diff = cur_box - prev_box
                score_diff = cur_score - prev_score
                for j in range(1, interval):
                    interp_frame = prev_frame + j
                    interp_box = prev_box + (box_diff * j / interval)  # interpolation
                    interp_score = prev_score + (score_diff * j / interval)
                    if interp_frame not in candidates_dict:
                        candidates_dict[interp_frame] = [
                            (trk_id, interp_box, interp_score)
                        ]
                    else:
                        candidates_dict[interp_frame].append(
                            (trk_id, interp_box, interp_score)
                        )
            prev_frame = cur_frame
            prev_box = cur_box

    if visualize:
        visualize_trajectories([trajectories[tid][0] for tid in vis_ids])
    for frame, candidates in candidates_dict.items():
        cand_ids = np.array([x[0] for x in candidates])
        cand_boxes = np.array([x[1] for x in candidates], dtype=np.float32).reshape(
            -1, 7
        )
        cand_scores = np.array([x[2] for x in candidates])

        if ignore_thresh > 0:
            # deletes the candidates which can be matched with the existing boxes in the same frame
            if frame in existing_boxes:
                ex_boxes = np.array(existing_boxes[frame], dtype=np.float32).reshape(
                    -1, 7
                )
                dis_aff_matrix = compute_ncd_matrix(cand_boxes, ex_boxes)
                valid_mask = np.max(dis_aff_matrix, axis=1) < ignore_thresh
                cand_ids = cand_ids[valid_mask]
                cand_boxes = cand_boxes[valid_mask]
                cand_scores = cand_scores[valid_mask]
            else:
                ex_boxes = np.empty((0, 7), dtype=np.float32)

        if score_thresh != 0:
            valid_mask = cand_scores >= score_thresh
            cand_scores = cand_scores[valid_mask]
            cand_ids = cand_ids[valid_mask]
            cand_boxes = cand_boxes[valid_mask]

        # NMS
        if nms_thresh != 0:
            order = np.argsort(cand_scores)
            nms_ids = []
            nms_boxes = []
            nms_scores = []
            while len(order) > 0:
                picked_box = cand_boxes[order[-1]]

                nms_ids.append(cand_ids[order[-1]])
                nms_boxes.append(picked_box)
                nms_scores.append(cand_scores[order[-1]])

                order = order[:-1]
                dis_aff_matrix = compute_ncd_matrix(
                    picked_box[np.newaxis], cand_boxes[order]
                )
                order = order[dis_aff_matrix[0] < nms_thresh]
            cand_ids = np.array(nms_ids)
            cand_boxes = np.array(nms_boxes)
            cand_scores = np.array(nms_scores)

        for cand_id, cand_box, cand_score in zip(cand_ids, cand_boxes, cand_scores):
            boxes, objs = trajectories[cand_id]
            if isinstance(boxes, list):
                boxes.append(cand_box)
            else:
                boxes = np.concatenate((boxes, cand_box[np.newaxis]))
            objs.append(
                NuscenesObject().from_box(
                    cand_box,
                    sample_token=frame,
                    velocity=(0, 0),
                    tracking_id=cand_id,
                    tracking_name=objs[0].data["tracking_name"],
                    tracking_score=cand_score,
                )
            )
            trajectories[cand_id] = sort_trajectory_by_frame((boxes, objs))

    if visualize:
        visualize_trajectories([trajectories[tid][0] for tid in vis_ids])

    return trajectories
