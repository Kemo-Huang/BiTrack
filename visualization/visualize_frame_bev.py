from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from utils import (Calibration, boxes_to_corners_bev, draw_boxes_bev,
                   get_lidar_boxes_from_objs, get_objects_from_label,
                   map_tracks_by_frames)


def main():
    parser = ArgumentParser()
    parser.add_argument('tag', type=str)
    parser.add_argument('split', type=str)
    parser.add_argument('seq', type=int)
    parser.add_argument('frame', type=int)
    args = parser.parse_args()
    trk_out_dir = Path(f'output/kitti/{args.split}/{args.tag}/data')
    gt_dir = Path(f'data/kitti/tracking/{args.split}/label_02')
    calib_dir = Path(f'data/kitti/tracking/{args.split}/calib')
    lidar_dir = Path(f'data/kitti/tracking/{args.split}/velodyne')
    assert trk_out_dir.exists()

    calib = Calibration(calib_dir / f'{str(args.seq).zfill(4)}.txt')
    seq_out_tracks = map_tracks_by_frames(
        get_objects_from_label(trk_out_dir / f'{str(args.seq).zfill(4)}.txt', track=True)
    )
    out_tracks = seq_out_tracks.get(args.frame, [])
    out_boxes_bev = boxes_to_corners_bev(get_lidar_boxes_from_objs(out_tracks, calib))
    out_ids = [x.tracking_id for x in out_tracks]

    seq_gt_tracks = map_tracks_by_frames([
        x for x in get_objects_from_label(gt_dir / f'{str(args.seq).zfill(4)}.txt', track=True) if x.cls_type == 'Car' or x.cls_type == 'Van'
    ])
    gt_tracks = seq_gt_tracks.get(args.frame, [])
    gt_boxes_bev = boxes_to_corners_bev(get_lidar_boxes_from_objs(gt_tracks, calib))
    gt_ids = [x.tracking_id for x in gt_tracks]

    points = np.fromfile(lidar_dir / str(args.seq).zfill(4) / f'{str(args.frame).zfill(6)}.bin', dtype=np.float32).reshape(-1, 4)[:, :2]

    fig, ax = plt.subplots()
    ax.axis('equal')

    draw_boxes_bev(ax, gt_boxes_bev, 'r', gt_ids)
    draw_boxes_bev(ax, out_boxes_bev, 'b', out_ids)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.scatter(points[:,0], points[:,1], s=1, c='gray')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plt.show()

if __name__ == '__main__':
    main()