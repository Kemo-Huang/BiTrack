from pathlib import Path

import numpy as np

from utils import Calibration, KittiTrack3d, get_objects_from_label


def main():
    max_distance = 50
    # original_label_dir = Path('data/kitti/tracking/training/label_02')
    original_label_dir = Path(
        "F://Github/3D-Multi-Object-Tracker/evaluation/results/virconv/data"
    )
    calib_dir = Path("data/kitti/tracking/training/calib")
    # my_label_dir = Path(f'data/kitti/tracking/training/label_near_{max_distance}')
    my_label_dir = Path(
        f"F://Github/3D-Multi-Object-Tracker/evaluation/results/virconv_{max_distance}/data"
    )
    my_label_dir.mkdir(exist_ok=True, parents=True)
    for seq in range(21):
        seq = str(seq).zfill(4)
        calib = Calibration(calib_dir / f"{seq}.txt")
        tracks = get_objects_from_label(f"{original_label_dir / seq}.txt", track=True)
        lines = []
        for track in tracks:
            track: KittiTrack3d
            box = track.to_lidar_box(calib)
            if np.linalg.norm(box[:2]) <= max_distance:
                lines.append(track.serialize())
        with open(my_label_dir / f"{seq}.txt", "w") as f:
            f.writelines(lines)


if __name__ == "__main__":
    main()
