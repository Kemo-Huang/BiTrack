import pickle
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment

from utils import (Calibration, boxes_to_corners_bev, compute_iou_2d,
                   draw_boxes_bev, get_global_boxes_from_lidar,
                   get_lidar_boxes_from_objs, get_objects_from_label,
                   get_poses_from_file, map_tracks_by_frames)


def main():
    parser = ArgumentParser()
    parser.add_argument('split', type=str)
    parser.add_argument('tag', type=str)
    parser.add_argument('det3d', type=str)
    args = parser.parse_args()
    trk_out_dir = Path(f'output/kitti/{args.split}/{args.tag}/data')
    gt_dir = Path(f'data/kitti/tracking/{args.split}/label_02')
    calib_dir = Path(f'data/kitti/tracking/{args.split}/calib')
    oxts_dir = Path(f'data/kitti/tracking/{args.split}/oxts')
    img_dir = Path(f'data/kitti/tracking/{args.split}/image_02')
    det_out_dir = Path(f'data/kitti/tracking/{args.split}/det3d_out/{args.det3d}')

    use_poses = True
    use_dets = True  # use detection results instead of tracking results
    show_frame_fn = False

    iou_thres = 0.5
    max_truncation = 0
    max_occlusion = 2
    min_height = 25

    with open(f'data/kitti/tracking/{args.split}/evaluate_tracking.seqmap.training') as f:
        lines = f.readlines()

    seq_frame_dict = {}
    for line in lines:
        split = line.split()
        seq_frame_dict[split[0]] = split[2:]

    for seq, frames in seq_frame_dict.items():
        tp = 0
        fp = 0
        fn = 0
        det = 0
        gt = 0
        if use_dets:
            seq_out_tracks = {int(frame.stem): get_objects_from_label(frame) for frame in (det_out_dir / seq).iterdir() if frame.suffix == '.txt'}
        else:
            seq_out_tracks = map_tracks_by_frames(
                get_objects_from_label(trk_out_dir / f'{seq}.txt', track=True))
        seq_gt_tracks = get_objects_from_label(
            gt_dir / f'{seq}.txt', track=True)
        seq_ignore_tracks = map_tracks_by_frames(
            [trk for trk in seq_gt_tracks if trk.cls_type == 'DontCare'])
        seq_distractor_tracks = map_tracks_by_frames(
            [trk for trk in seq_gt_tracks if trk.cls_type == 'Van'])
        seq_gt_tracks = map_tracks_by_frames(
            [trk for trk in seq_gt_tracks if trk.cls_type == 'Car'])

        calib = Calibration(calib_dir / f'{seq}.txt')
        poses = get_poses_from_file(
            oxts_dir / f'{seq}.txt') if use_poses else None

        fig, ax = plt.subplots()
        ax.axis('equal')

        seq_tp_boxes = []
        seq_fp_boxes = []
        seq_fn_boxes = []
        seq_invalid_boxes = []

        start_frame, end_frame = frames
        start_ids = set()
        start_centers = []
        for frame in range(int(start_frame), int(end_frame)):
            out_tracks = seq_out_tracks.get(frame, [])
            gt_tracks = seq_gt_tracks.get(frame, [])
            dis_tracks = seq_distractor_tracks.get(frame, [])
            ignore_tracks = seq_ignore_tracks.get(frame, [])

            out_boxes_bev = boxes_to_corners_bev(
                get_global_boxes_from_lidar(get_lidar_boxes_from_objs(
                    out_tracks, calib), poses[frame]) if use_poses else get_lidar_boxes_from_objs(out_tracks, calib)
            )
            out_ids = None if use_dets else np.array([trk.tracking_id for trk in out_tracks])
            gt_boxes_bev = boxes_to_corners_bev(
                get_global_boxes_from_lidar(get_lidar_boxes_from_objs(
                    gt_tracks, calib), poses[frame]) if use_poses else get_lidar_boxes_from_objs(gt_tracks, calib)
            )
            gt_ids = np.array([trk.tracking_id for trk in gt_tracks])

            bboxes1 = np.stack([out_trk.box2d for out_trk in out_tracks]) if len(
                out_tracks) > 0 else np.empty((0, 4))
            bboxes2 = np.stack([dis_trk.box2d for dis_trk in dis_tracks]) if len(
                dis_tracks) > 0 else np.empty((0, 4))
            bboxes3 = np.stack([gt_trk.box2d for gt_trk in gt_tracks]) if len(
                gt_tracks) > 0 else np.empty((0, 4))
            bboxes4 = np.stack([ignore_trk.box2d for ignore_trk in ignore_tracks]) if len(
                ignore_tracks) > 0 else np.empty((0, 4))

            truncations = np.array(
                [0] * len(bboxes2) + [gt_trk.truncation for gt_trk in gt_tracks], dtype=int)
            occlusions = np.array(
                [0] * len(bboxes2) + [gt_trk.occlusion for gt_trk in gt_tracks], dtype=int)

            # preprocess
            ious = compute_iou_2d(bboxes1, np.concatenate((bboxes2, bboxes3)))
            ious[ious < iou_thres - np.finfo('float').eps] = 0

            row_inds, col_inds = linear_sum_assignment(ious, maximize=True)
            mask = ious[row_inds, col_inds] > 0 + np.finfo('float').eps
            row_inds = row_inds[mask]
            col_inds = col_inds[mask]

            to_remove_matched_mask = np.logical_or(
                np.isin(col_inds, np.arange(len(bboxes2))),
                np.logical_or(
                    truncations[col_inds] > max_truncation,
                    occlusions[col_inds] > max_occlusion
                )
            )
            to_remove_matched = row_inds[to_remove_matched_mask]

            unmatched_indices = np.delete(
                np.arange(len(bboxes1)), row_inds, axis=0)
            unmatched_bboxes1 = bboxes1[unmatched_indices]

            heights = unmatched_bboxes1[:, 3] - unmatched_bboxes1[:, 1]
            is_too_small = heights <= min_height + np.finfo('float').eps

            intersection_with_ignore_region = compute_iou_2d(
                unmatched_bboxes1, bboxes4, do_ioa=True)
            is_within_crowd_ignore_region = np.any(
                intersection_with_ignore_region > iou_thres + np.finfo('float').eps, axis=1)

            to_remove_unmatched = unmatched_indices[np.logical_or(
                is_too_small, is_within_crowd_ignore_region)]
            to_remove_inds = np.concatenate(
                (to_remove_matched, to_remove_unmatched))

            bboxes1 = np.delete(bboxes1, to_remove_inds, axis=0)
            invalid_out_boxes_bev = out_boxes_bev[to_remove_inds]
            out_boxes_bev = np.delete(out_boxes_bev, to_remove_inds, axis=0)
            if not use_dets:
                out_ids = np.delete(out_ids, to_remove_inds, axis=0)
                for idx in range(len(out_boxes_bev)):
                    if out_ids[idx] not in start_ids:
                        start_ids.add(out_ids[idx])
                        start_box = out_boxes_bev[idx]
                        start_centers.append((start_box[0] + start_box[2]) / 2)

            ious = np.delete(ious, to_remove_inds, axis=0)
            ious = ious[:, len(bboxes2):]

            to_keep_mask = np.logical_and(truncations[len(
                bboxes2):] <= max_truncation, occlusions[len(bboxes2):] <= max_occlusion)
            bboxes3 = bboxes3[to_keep_mask]
            ious = ious[:, to_keep_mask]
            gt_boxes_bev = gt_boxes_bev[to_keep_mask]
            gt_ids = gt_ids[to_keep_mask]

            row_inds, col_inds = linear_sum_assignment(ious, maximize=True)
            mask = ious[row_inds, col_inds] > 0 + np.finfo('float').eps
            row_inds = row_inds[mask]
            col_inds = col_inds[mask]

            matches = len(row_inds)
            tp_boxes_bev = out_boxes_bev[row_inds]
            fp_boxes_bev = out_boxes_bev[np.delete(
                np.arange(len(bboxes1)), row_inds)]
            fn_boxes_bev = gt_boxes_bev[np.delete(
                np.arange(len(bboxes3)), col_inds)]

            det += len(bboxes1)
            gt += len(bboxes3)
            tp += matches
            fp += len(bboxes1) - matches
            fn += len(bboxes3) - matches

            if (show_frame_fn and len(bboxes3) - matches > 0):
                tp_boxes = bboxes1[row_inds].reshape(-1, 4)
                fp_boxes = bboxes1[np.delete(np.arange(len(bboxes1)), row_inds)].reshape(-1, 4)
                fn_boxes = bboxes3[np.delete(np.arange(len(bboxes3)), col_inds)].reshape(-1, 4)
                im = cv2.imread(str(img_dir / seq / f'{str(frame).zfill(6)}.png'))
                tp_boxes = np.round(tp_boxes).astype(int)
                for tp_box in tp_boxes:
                    cv2.rectangle(im, (tp_box[0], tp_box[1]), (tp_box[2], tp_box[3]), color=(0, 255, 0))
                fp_boxes = np.round(fp_boxes).astype(int)
                for fp_box in fp_boxes:
                    cv2.rectangle(im, (fp_box[0], fp_box[1]), (fp_box[2], fp_box[3]), color=(255, 0, 0))
                fn_boxes = np.round(fn_boxes).astype(int)
                for fn_box in fn_boxes:
                    cv2.rectangle(im, (fn_box[0], fn_box[1]), (fn_box[2], fn_box[3]), color=(0, 0, 255))
                cv2.imshow(str(frame), im)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            seq_tp_boxes.append(tp_boxes_bev)
            seq_fp_boxes.append(fp_boxes_bev)
            seq_fn_boxes.append(fn_boxes_bev)
            seq_invalid_boxes.append(invalid_out_boxes_bev)

        seq_tp_boxes = np.concatenate(seq_tp_boxes)
        seq_fp_boxes = np.concatenate(seq_fp_boxes)
        seq_fn_boxes = np.concatenate(seq_fn_boxes)
        seq_invalid_boxes = np.concatenate(seq_invalid_boxes)

        draw_boxes_bev(ax, seq_invalid_boxes, 'gray')
        draw_boxes_bev(ax, seq_tp_boxes, 'g')
        draw_boxes_bev(ax, seq_fp_boxes, 'b')
        draw_boxes_bev(ax, seq_fn_boxes, 'r')
        if not use_dets:
            for center in start_centers:
                ax.scatter(center[0], center[1], color='black')

        print(
            f'seq {seq}, frames: {int(end_frame)}, det: {det}, gt: {gt}, tp: {tp}, fn: {fn}, fp: {fp}')
        plt.show()


if __name__ == '__main__':
    main()
