from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from utils import (Calibration, crop_points_from_boxes,
                   get_lidar_boxes_from_objs, get_objects_from_label,
                   points_inside_boxes)


def crop_points(label_dir: Path, lidar_dir: Path, calib: Calibration, out_dir: Path, is_gt=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(label_dir.name)
    for label_file in label_dir.iterdir():
        objs = get_objects_from_label(label_file, track=False)
        if is_gt:
            objs = [obj for obj in objs if obj.cls_type ==
                    'Car' or obj.cls_type == 'Van']

        frame = label_file.stem
        cur_out_dir = out_dir / frame
        if len(objs) > 0:
            cur_out_dir.mkdir(exist_ok=True)
            boxes = get_lidar_boxes_from_objs(objs, calib)
            inside_points = crop_points_from_boxes(
                np.fromfile(lidar_dir / f'{frame}.bin',
                            dtype=np.float32).reshape(-1, 4),
                boxes, front_only=True
            )
            for idx, cur_points in enumerate(inside_points):
                cur_points.tofile(
                    cur_out_dir / f'{idx}.bin')


def test(label_dir: Path, lidar_dir: Path, calib: Calibration, is_gt=False):
    for label_file in label_dir.iterdir():
        objs = get_objects_from_label(label_file, track=False)
        if is_gt:
            objs = [obj for obj in objs if obj.cls_type ==
                    'Car' or obj.cls_type == 'Van']
        if len(objs) > 0:
            boxes = get_lidar_boxes_from_objs(objs, calib)
            for i in range(len(boxes)):
                points = np.fromfile(
                    lidar_dir / label_file.stem / f'{i}.bin', dtype=np.float32).reshape(-1, 4)
                assert np.all(points_inside_boxes(boxes[i:i+1], points[:, :3]))

def mp_func(args):
    label_dir, lidar_dir, calib, out_dir, is_gt = args
    crop_points(label_dir, lidar_dir, calib, out_dir, is_gt)

def main():
    parser = ArgumentParser()
    parser.add_argument('split', type=str)
    parser.add_argument('det_name', type=str)
    args = parser.parse_args()

    lidar_dir = Path(f'data/kitti/tracking/{args.split}/velodyne')
    out_dir = Path(f'data/kitti/tracking/{args.split}/cropped_points')
    detection_dir = Path(f'data/kitti/tracking/{args.split}/det3d_out/{args.det_name}')
    calib_dir = Path(f'data/kitti/tracking/{args.split}/calib')

    is_gt = False

    with Pool(8) as p:
        p.map(mp_func, [(detection_dir / seq.name, seq, Calibration(calib_dir / f'{seq.name}.txt'), out_dir / args.det_name / seq.name, is_gt) for seq in lidar_dir.iterdir()])

if __name__ == '__main__':
    main()
