import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from torch.utils.data import Dataset

from suscape_utils import SuscapeObject
from utils import get_frustum_points, rotate_points_bev


def collate_fn(batch_list: list, _unused=False):
    cam_data_dict = {cam: defaultdict(list) for cam in SuscapeDataset.cameras}
    ret = {cam: {} for cam in SuscapeDataset.cameras}
    batch_size = len(batch_list)
    for batch_cam_dict in batch_list:
        for cam, cur_sample in batch_cam_dict.items():
            for key, val in cur_sample.items():
                cam_data_dict[cam][key].append(val)
    for cam, data_dict in cam_data_dict.items():
        for key, val in data_dict.items():
            if key in ['voxels', 'voxel_num_points']:
                ret[cam][key] = np.concatenate(val, axis=0)
            elif key in ['points', 'voxel_coords']:
                coors = []
                for i, coor in enumerate(val):
                    coor_pad = np.pad(coor, ((0, 0), (1, 0)),
                                    mode='constant', constant_values=i)
                    coors.append(coor_pad)
                ret[cam][key] = np.concatenate(coors, axis=0)
            elif key in ['frame_id']:
                ret[cam][key] = val
            else:
                ret[cam][key] = np.stack(val, axis=0)

    ret['batch_size'] = batch_size
    return ret


def format_suscape_detection_results(pred_dicts, class_names, frame_ids) -> Dict[List[Dict]]:
    res = {frame_id: [] for frame_id in frame_ids}
    for batch_idx, box_dict in enumerate(pred_dicts):
        frame_id = frame_ids[batch_idx]
        pred_scores = box_dict['pred_scores'].cpu().numpy()
        pred_boxes = box_dict['pred_boxes'].cpu().numpy()
        pred_labels = box_dict['pred_labels'].cpu().numpy()
        num_samples = len(pred_scores)

        if num_samples > 0:
            names = np.array(class_names)[pred_labels - 1]

            for i in range(num_samples):
                obj = SuscapeObject(
                    0, names[i], pred_boxes[i], pred_scores[i])
                res[frame_id].append(obj.serialize())
    return res


def write_suscape_detection_results_to_json(res):
    for cam in res:
        angle = SuscapeDataset.angles[cam]
        


class SuscapeDataset(Dataset):
    cameras = ['front', 'front_left', 'front_right', 'rear', 'rear_left', 'rear_right']
    angles = {
        'front': 0,
        'front_left': np.pi / 3,
        'front_right': -np.pi / 3,
        'rear': np.pi,
        'rear_left': np.pi * 2 / 3,
        'rear_right': -np.pi * 2 / 3
    }
    def __init__(
        self,
        root_path: Path,
        voxel_generator,
        point_cloud_range,
        aug_cfg
    ):
        super().__init__()
        self.root_path = root_path
        self.voxel_generator = voxel_generator
        self.point_cloud_range = point_cloud_range
        self.aug_cfg = aug_cfg
        self.sample_ids = [f.stem for f in (self.root_path / 'lidar_bin').iterdir()]

        self.calibs = {}
        for cam in self.cameras:
            with open(self.root_path / 'calib' / 'camera' / f'{cam}.json') as f:
                calib = json.load(f)
            intrinsic = np.array(calib['intrinsic']) # (3, 3)
            extrinsic = np.array(calib['extrinsic'])[:3, :]  # (3, 4)
            lidar2img = np.dot(intrinsic, extrinsic)  # (3, 4)
            self.calibs[cam] = lidar2img
    
    def get_lidar(self, idx):
        return np.fromfile(self.root_path / 'lidar_bin' / f'{self.sample_ids[idx]}.bin', dtype=np.float32).reshape(-1, 4)

    @staticmethod
    def convert_coords_to_x_foward(points):
        """Original coord: x leftward, y backward, z upward. Convert to x forward, y leftward, z upward.
        """
        points[:, [0, 1]] = points[:, [1, 0]]
        points[:, 0] *= -1
        return points

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        all_points = self.get_lidar(idx)
        all_points = self.convert_coords_to_x_foward(all_points)
        all_data_dict = {}

        for cam, lidar2img in self.calibs.items():
            _, mask = get_frustum_points(all_points[:, :3], lidar2img, 1536, 2048)

            points = np.copy(all_points[mask])

            # convert to x forward
            points = rotate_points_bev(points[np.newaxis], -np.stack((self.angles[cam])))[0]

            if self.aug_cfg is not None:
                aug_method, aug_param = self.aug_cfg
                if aug_method == 'scale':
                    points[:, :3] *= aug_param
                elif aug_method == 'rotate':
                    points = rotate_points_bev(points[np.newaxis], np.array(aug_param)[np.newaxis])[0]
                elif aug_method == 'translate':
                    points[:, :3] += aug_param
                elif aug_method == 'flip':
                    points[:, 1] *= -1

            mask = np.all(np.logical_and(
                points[:, :3] >= self.point_cloud_range[:3],
                points[:, :3] <= self.point_cloud_range[3:6]
            ), axis=1)
            points = points[mask]

            voxels, coordinates, num_points = self.voxel_generator.generate(points)
            data_dict = {
                'frame_id': self.sample_ids[idx],
                'voxels': voxels,
                'voxel_coords': coordinates,
                'voxel_num_points': num_points
            }
            all_data_dict[cam] = data_dict

        return data_dict
