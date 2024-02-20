import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from utils import (Calibration, KittiObject3d, get_frustum_points,
                   rotate_points_bev)


def collate_fn(batch_list, _unused=False):
    data_dict = defaultdict(list)
    for cur_sample in batch_list:
        for key, val in cur_sample.items():
            data_dict[key].append(val)
    batch_size = len(batch_list)
    ret = {}

    for key, val in data_dict.items():
        if key in ["voxels", "voxel_num_points"]:
            ret[key] = np.concatenate(val, axis=0)
        elif key in ["points", "voxel_coords"]:
            coors = []
            for i, coor in enumerate(val):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)), mode="constant", constant_values=i
                )
                coors.append(coor_pad)
            ret[key] = np.concatenate(coors, axis=0)
        elif key in ["frame_id"]:
            ret[key] = val
        else:
            ret[key] = np.stack(val, axis=0)

    ret["batch_size"] = batch_size
    return ret


def write_kitti_detection_results(
    pred_dicts, class_names, frame_ids, calibrations, img_hw_dict, output_dir: Path
):
    output_dir.mkdir(exist_ok=True, parents=True)
    for batch_idx, box_dict in enumerate(pred_dicts):
        frame_id = frame_ids[batch_idx]
        pred_scores = box_dict["pred_scores"].cpu().numpy()
        pred_boxes = box_dict["pred_boxes"].cpu().numpy()
        pred_labels = box_dict["pred_labels"].cpu().numpy()
        num_samples = len(pred_scores)

        lines = []
        if num_samples > 0:
            calib = (
                calibrations
                if isinstance(calibrations, Calibration)
                else calibrations[batch_idx]
            )
            names = np.array(class_names)[pred_labels - 1]

            for i in range(num_samples):
                try:
                    obj = KittiObject3d(
                        img_hw=img_hw_dict[str(frame_id).zfill(6)]
                    ).from_lidar_box(pred_boxes[i], calib, names[i], pred_scores[i])
                    line = obj.serialize()
                    lines.append(line)
                except ValueError:
                    pass

        cur_det_file = output_dir / ("%s.txt" % frame_id)
        with open(cur_det_file, "w") as f:
            f.writelines(lines)


class KittiDataset(Dataset):
    def __init__(
        self, root_path: Path, seq, voxel_generator, point_cloud_range, aug_cfg
    ):
        super().__init__()
        self.root_path = root_path
        if isinstance(seq, int):
            seq = str(seq).zfill(4)
        self.seq = seq
        self.voxel_generator = voxel_generator
        self.point_cloud_range = point_cloud_range
        self.aug_cfg = aug_cfg
        self.sample_ids = [
            f.stem for f in (self.root_path / "velodyne" / self.seq).iterdir()
        ]

        self.calib = Calibration(self.root_path / "calib" / f"{self.seq}.txt")
        self.img_hw_dict = json.load(open(self.root_path / "img_hw.json"))[seq]

    def get_lidar(self, idx):
        return np.fromfile(
            self.root_path / "velodyne" / self.seq / f"{self.sample_ids[idx]}.bin",
            dtype=np.float32,
        ).reshape(-1, 4)

    def get_image_height_width(self, idx):
        # image = Image.open(self.root_path / 'image_02' /
        #                    self.seq / f'{self.sample_ids[idx]}.png')
        # return image.height, image.width
        return self.img_hw_dict[str(self.sample_ids[idx]).zfill(6)]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        points = self.get_lidar(idx)
        h, w = self.get_image_height_width(idx)

        proj, mask = get_frustum_points(points[:, :3], self.calib, h, w, kitti=True)
        points = points[mask]

        if self.aug_cfg is not None:
            aug_method, aug_param = self.aug_cfg
            if aug_method == "scale":
                points[:, :3] *= aug_param
            elif aug_method == "rotate":
                points = rotate_points_bev(
                    points[np.newaxis], np.array(aug_param)[np.newaxis]
                )[0]
            elif aug_method == "translate":
                points[:, :3] += aug_param
            elif aug_method == "flip":
                points[:, 1] *= -1

        mask = np.all(
            np.logical_and(
                points[:, :3] >= self.point_cloud_range[:3],
                points[:, :3] <= self.point_cloud_range[3:6],
            ),
            axis=1,
        )
        points = points[mask]

        voxels, coordinates, num_points = self.voxel_generator.generate(points)
        data_dict = {
            "frame_id": self.sample_ids[idx],
            "voxels": voxels,
            "voxel_coords": coordinates,
            "voxel_num_points": num_points,
        }

        return data_dict
