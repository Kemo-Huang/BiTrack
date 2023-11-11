from pathlib import Path

import numpy as np

from utils import (Calibration, get_global_boxes_from_lidar,
                   get_poses_from_file, read_kitti_trajectories_from_file,
                   visualize_trajectories)


def main():
    seq = "0006"
    root_dir = Path("data/kitti/tracking/training")
    calib = Calibration(root_dir / f"calib/{seq}.txt")
    poses = get_poses_from_file(root_dir / f"oxts/{seq}.txt")
    tracks = read_kitti_trajectories_from_file(
        seq, Path("output/kitti/training/bitrack/data"), calib
    )
    all_boxes = []
    for boxes, objs in tracks.values():
        global_boxes = []
        for box, obj in zip(boxes, objs):
            frame = int(obj.sample_id)
            if frame > 100:
                continue
            global_boxes.append(
                get_global_boxes_from_lidar(box[np.newaxis], poses[frame])[0]
            )
        all_boxes.append(np.array(global_boxes).reshape(-1, 7))
    visualize_trajectories(all_boxes)


if __name__ == "__main__":
    main()
