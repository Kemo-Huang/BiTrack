from typing import List, Tuple

import numpy as np
import torch
from rtree import index
from shapely.geometry import Polygon

from detection.voxel_rcnn.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
from utils import angle_in_range, boxes_to_corners_3d, boxes_to_corners_bev

from .association import Matcher
from .detections import Detections
from .track import Track


def compute_3d_iou_matrix(bboxes1: np.ndarray, bboxes2: np.ndarray):
    a = torch.from_numpy(bboxes1).float().to("cuda:0")
    b = torch.from_numpy(bboxes2).float().to("cuda:0")
    iou = boxes_iou3d_gpu(a, b).cpu().numpy()
    return iou


def compute_3d_iou_matrix_shapely(boxes_a, boxes_b):
    m = len(boxes_a)
    n = len(boxes_b)
    if m > n:
        # More memory-efficient to compute it the other way round and
        # transpose.
        return compute_3d_iou_matrix_shapely(boxes_b, boxes_a).T

    polys_a = [
        Polygon(np.concatenate((corners, corners[0:1]), axis=0))
        for corners in boxes_to_corners_bev(boxes_a)
    ]
    polys_b = [
        Polygon(np.concatenate((corners, corners[0:1]), axis=0))
        for corners in boxes_to_corners_bev(boxes_b)
    ]

    # Build a spatial index for rects_a.
    index_a = index.Index()
    for i, a in enumerate(polys_a):
        index_a.insert(i, a.bounds)

    # Find candidate intersections using the spatial index.
    iou = np.zeros((m, n))
    for j, b in enumerate(polys_b):
        for i in index_a.intersection(b.bounds):
            a = polys_a[i]
            intersection_bev = a.intersection(b).area
            if intersection_bev:
                a_height_max = boxes_a[i, 2] + boxes_a[i, 5] / 2
                a_height_min = boxes_a[i, 2] - boxes_a[i, 5] / 2
                b_height_max = boxes_b[j, 2] + boxes_b[j, 5] / 2
                b_height_min = boxes_b[j, 2] - boxes_b[j, 5] / 2

                max_of_min = max(a_height_min, b_height_min)
                min_of_max = min(a_height_max, b_height_max)
                overlaps_h = max(min_of_max - max_of_min, 0)

                intersection_3d = intersection_bev * overlaps_h

                vol_a = boxes_a[i, 3] * boxes_a[i, 4] * boxes_a[i, 5]
                vol_b = boxes_b[j, 3] * boxes_b[j, 4] * boxes_b[j, 5]

                iou[i, j] = intersection_3d / max(vol_a + vol_b - intersection_3d, 1e-6)

    return iou


def compute_center_dis_matrix(det_boxes, pred_boxes):
    center_dis_matrix = np.linalg.norm(
        det_boxes[:, np.newaxis, :3] - pred_boxes[np.newaxis, :, :3], ord=2, axis=-1
    )
    return center_dis_matrix


def compute_ncd_matrix(det_boxes: np.ndarray, pred_boxes: np.ndarray):
    center_dis_matrix = compute_center_dis_matrix(det_boxes, pred_boxes)
    vertex_dis_matrix = np.max(
        np.linalg.norm(
            boxes_to_corners_3d(det_boxes)[:, np.newaxis, :, np.newaxis, :]
            - boxes_to_corners_3d(pred_boxes)[np.newaxis, :, np.newaxis, :, :],
            ord=2,
            axis=-1,
        ).reshape(len(det_boxes), len(pred_boxes), 64),
        axis=-1,
    )

    ncd_matrix = center_dis_matrix / vertex_dis_matrix  # [0, 1)
    return 1 - ncd_matrix  # (0, 1]


def compute_angle_matrix(det_boxes: np.ndarray, pred_boxes: np.ndarray):
    for i in range(len(det_boxes)):
        det_boxes[i, 6] = angle_in_range(det_boxes[i, 6])
    diff = np.abs(det_boxes[:, np.newaxis, 6] - pred_boxes[np.newaxis, :, 6])
    assert np.all(diff <= 2 * np.pi), f"{det_boxes[:, 6], pred_boxes[:, 6]}"
    mask = diff > np.pi
    diff[mask] = 2 * np.pi - diff[mask]
    mask = diff > np.pi / 2
    diff[mask] = np.pi - diff[mask]
    return 1 / 2 - diff / np.pi  # [0, 1]


def compute_box_matrix(det_boxes: np.ndarray, pred_boxes: np.ndarray):
    return -np.linalg.norm(det_boxes - pred_boxes)


def compute_app_matrix(a: np.ndarray, b: np.ndarray, is_normalized=False):
    """Cosine similarity matrix"""
    if not is_normalized:
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        b /= np.linalg.norm(b, axis=1, keepdims=True)
    app_matrix = np.dot(a, b.T)

    return app_matrix


class Tracker:
    def __init__(
        self,
        t_miss=10,
        t_miss_new=3,
        t_hit=2,
        match_algorithm=1,
        aff_thresh=0.5,
        ang_thresh=0.5,
        app_thresh=0.5,
        ent_ex_score=0.3,
        offline=True,
        app_m=0.9,
        p=10,
        q=2,
        ang_vel=True,
        vel_reinit=True,
        sim_metric="NCD",
    ):
        self.t_miss = t_miss
        self.t_miss_new = t_miss_new
        self.t_hit = t_hit

        self.tracks: List[Track] = []
        self.frame_counter = 0
        self.matcher = Matcher(match_algorithm)
        self.aff_thresh = aff_thresh
        self.ang_thresh = ang_thresh
        self.app_thresh = app_thresh
        self.app_m = app_m
        self.ent_ex_score = ent_ex_score
        self.offline = offline
        self.p = p
        self.q = q
        self.ang_vel = ang_vel
        self.vel_reinit = vel_reinit
        self.sim_metric = sim_metric

    def reset(self):
        self.tracks = []
        self.frame_counter = 0
        Track.global_cur_id = 1

    def optimize(
        self,
        aff_matrix,
        aff_thresh=0,
        det_scores=None,
        trk_scores=None,
        post_valid_mask=None,
        unused=None,
    ):
        if self.matcher.algorithm == "HA" or det_scores is None:
            # HA matching
            return self.matcher.match(
                aff_matrix, aff_thresh, post_valid_mask=post_valid_mask, unused=unused
            )

        elif self.matcher.algorithm == "MCF":
            # MCF matching: also use detection confidences and entry/exit scores
            assert len(det_scores) == aff_matrix.shape[0]
            assert len(trk_scores) == aff_matrix.shape[1]
            return self.matcher.match(
                aff_matrix=aff_matrix,
                det_scores=det_scores,
                trk_scores=trk_scores,
                entry_scores=self.ent_ex_score * np.ones(aff_matrix.shape[0]),
                exit_scores=self.ent_ex_score * np.ones(aff_matrix.shape[1]),
            )
        else:
            raise NotImplementedError

    def aff_match(
        self,
        det_boxes,
        pred_boxes,
        det_embeds=None,
        match_matrix=None,
        entry_vec=None,
        exit_vec=None,
        false_det_vec=None,
        false_trk_vec=None,
        det_scores=None,
        trk_scores=None,
        det_valid_mask=None,
        trk_valid_mask=None,
    ):
        if self.sim_metric == "IoU":
            aff_matrix = compute_3d_iou_matrix(det_boxes, pred_boxes)
        elif self.sim_metric == "CD":
            aff_matrix = -compute_center_dis_matrix(det_boxes, pred_boxes)
        elif self.sim_metric == "NCD":
            aff_matrix = compute_ncd_matrix(det_boxes, pred_boxes)
        else:
            raise NotImplementedError

        # uses angle and appearance as post gates
        if self.ang_thresh > 0:
            angle_matrix = compute_angle_matrix(det_boxes, pred_boxes)
            post_valid_mask = angle_matrix >= self.ang_thresh
        else:
            angle_matrix = None
            post_valid_mask = np.ones_like(aff_matrix, dtype=bool)

        if det_embeds is None:
            app_matrix = None
        else:
            trk_embeds = np.stack([trk.embed for trk in self.tracks])
            app_matrix = compute_app_matrix(det_embeds, trk_embeds, is_normalized=True)
            post_valid_mask &= app_matrix >= self.app_thresh

        if det_valid_mask is None or trk_valid_mask is None:
            results = self.optimize(
                aff_matrix,
                self.aff_thresh,
                det_scores,
                trk_scores,
                post_valid_mask,
                unused=app_matrix,
            )
            return results

        else:
            valid_match_ind = np.ix_(det_valid_mask, trk_valid_mask)
            aff_matrix = aff_matrix[valid_match_ind]
            post_valid_mask = post_valid_mask[valid_match_ind]
            if app_matrix is not None:
                app_matrix = app_matrix[valid_match_ind]

            if self.matcher.algorithm == "MCF" and det_scores is not None:
                det_scores = det_scores[det_valid_mask]
                trk_scores = trk_scores[trk_valid_mask]

            results = self.optimize(
                aff_matrix,
                self.aff_thresh,
                det_scores,
                trk_scores,
                post_valid_mask,
                unused=angle_matrix,
            )
            match_matrix[valid_match_ind] = results[0]
            entry_vec[det_valid_mask] = results[1]
            exit_vec[trk_valid_mask] = results[2]
            false_det_vec[det_valid_mask] = results[3]
            false_trk_vec[trk_valid_mask] = results[4]

            return match_matrix, entry_vec, exit_vec, false_det_vec, false_trk_vec

    def predict(self, num_passed_frames=1) -> Tuple[np.ndarray, np.ndarray]:
        self.frame_counter += num_passed_frames  # used for track management
        if len(self.tracks) > 0:
            ids = np.array([trk.id for trk in self.tracks], dtype=int)
            pred_boxes = np.stack(
                [trk.predict(num_passed_frames) for trk in self.tracks]
            )
            # should keep the order as the same as self.tracks
            pred_boxes.flags.writeable = False
            for trk in self.tracks:
                trk.misses += num_passed_frames
        else:
            ids = np.array([], dtype=int)
            pred_boxes = np.empty((0, 7), dtype=np.float32)
        return ids, pred_boxes

    def associate(self, pred_boxes: np.ndarray, detections: Detections = None):
        if detections is None:
            det_boxes = []
            det_embeds = None
            objs = None
        else:
            det_boxes = detections.boxes3d
            det_embeds = detections.embeds
            objs = detections.objs

        num_dets = len(det_boxes)
        num_trks = len(pred_boxes)

        if num_dets > 0 and num_trks > 0:
            if isinstance(det_boxes, list):
                det_boxes = np.stack(det_boxes)
            if isinstance(pred_boxes, list):
                pred_boxes = np.stack(pred_boxes)
            if objs is not None:
                assert num_dets == len(objs)
                det_scores = np.array([meta.tracking_score for meta in objs])
                trk_scores = np.array([trk.obj.tracking_score for trk in self.tracks])
            else:
                det_scores = None
                trk_scores = None

            match_matrix = np.zeros((num_dets, num_trks), dtype=bool)
            entry_vec = np.zeros(num_dets, dtype=bool)
            exit_vec = np.zeros(num_trks, dtype=bool)
            false_det_vec = np.zeros(num_dets, dtype=bool)
            false_trk_vec = np.zeros(num_trks, dtype=bool)
            det_valid_mask = np.ones(num_dets, dtype=bool)
            trk_valid_mask = np.ones(num_trks, dtype=bool)

            (
                match_matrix,
                entry_vec,
                exit_vec,
                false_det_vec,
                false_trk_vec,
            ) = self.aff_match(
                det_boxes,
                pred_boxes,
                det_embeds,
                match_matrix,
                entry_vec,
                exit_vec,
                false_det_vec,
                false_trk_vec,
                det_scores,
                trk_scores,
                det_valid_mask,
                trk_valid_mask,
            )

            matched = np.stack(np.nonzero(match_matrix)).T  # (N, 2)
            entry_dets = np.nonzero(entry_vec)[0]
            exit_trks = np.nonzero(exit_vec)[0]
            false_trks = np.nonzero(false_trk_vec)[0]

        else:
            if num_dets == 0 and num_trks > 0:
                # exit all tracks
                entry_dets = np.array([], dtype=int)
                exit_trks = np.arange(num_trks, dtype=int)
            else:
                # init all detections as tracks
                entry_dets = np.arange(num_dets, dtype=int)
                exit_trks = np.array([], dtype=int)

            matched = np.empty((0, 2), dtype=int)
            false_trks = np.array([], dtype=int)

        return matched, entry_dets, exit_trks, false_trks

    def update(
        self, matched, entry_dets, exit_trks, false_trks, detections: Detections
    ):
        if detections is None:
            det_boxes = []
            det_embeds = None
            objs = None
        else:
            det_boxes = detections.boxes3d
            det_embeds = detections.embeds
            objs = detections.objs

        # matched pairs
        for d, t in matched:
            self.tracks[t].misses = 0
            self.tracks[t].hits += 1
            self.tracks[t].new = False
            if self.offline:
                self.tracks[t].max_hits = max(
                    self.tracks[t].hits, self.tracks[t].max_hits
                )
            self.tracks[t].update(
                det_boxes[d],
                obj=None if objs is None else objs[d],
                embed=None if det_embeds is None else det_embeds[d],
            )

        # exit
        for i in exit_trks:
            # misses are incremented in predict()
            self.tracks[i].hits = 0

        # false
        for i in sorted(false_trks, reverse=True):
            self.tracks.pop(i)

        # entry
        for i in entry_dets:
            self.tracks.append(
                Track(
                    det_boxes[i],
                    obj=None if objs is None else objs[i],
                    embed=None if det_embeds is None else det_embeds[i],
                    offline=self.offline,
                    momentum=self.app_m,
                    p=self.p,
                    q=self.q,
                    ang_vel=self.ang_vel,
                    vel_reinit=self.vel_reinit,
                )
            )  # hits == misses = 0

    def track_management(self) -> Tuple[List[Track], List[Track]]:
        online_tracks = []
        dead_tracks = []
        # Uses reversed-order 'pop' for faster computation
        idx = len(self.tracks)
        for trk in reversed(self.tracks):
            idx -= 1
            if trk.misses >= self.t_miss or (trk.new and trk.misses >= self.t_miss_new):
                # Removes dead tracks
                dead_tracks.append(self.tracks.pop(idx))
            elif trk.misses == 0 and (
                self.frame_counter <= self.t_hit or trk.hits >= self.t_hit
            ):
                # Reports data of matched tracks
                online_tracks.append(trk)
        return online_tracks, dead_tracks
