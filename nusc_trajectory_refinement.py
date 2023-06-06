import json
import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List

import numpy as np
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits

from tracking.trajectory_clustering_split_and_recombination import \
    merge_forward_backward_trajectories


def load_trajectories(objs: List[List[Dict]], scene_box_dir: Path, timestamps: List):
    trajectories = {}

    for boxes_file in scene_box_dir.iterdir():
        track_id = int(boxes_file.stem)
        boxes = np.fromfile(boxes_file, dtype=np.float32).reshape(-1, 7)
        boxes = [box for box in boxes]
        trajectories[track_id] = [boxes, []]
    for sample_objs, timestamp in zip(objs, timestamps):
        for obj_dict in sample_objs:
            track = TrackingBox.deserialize(obj_dict)
            track.sample_id = timestamp
            trajectories[track.tracking_id][1].append(track)

    return trajectories


def main():
    merge = True

    # ============================================================================

    parser = ArgumentParser()
    parser.add_argument('out_tag', type=str)
    parser.add_argument('--forward_tag', type=str, default='forward')
    parser.add_argument('--backward_tag', type=str, default='backward')
    args = parser.parse_args()

    out_dir = Path(f'output/nusc/{args.out_tag}')
    out_json = out_dir / 'results.json'
    out_numpy_dir = out_dir / 'numpy'
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    out_numpy_dir.mkdir()
    forward_dir = Path(f'output/nusc/{args.forward_tag}')
    backward_dir = Path(f'output/nusc/{args.backward_tag}')
    forward_json = forward_dir / 'results.json'
    backward_json = backward_dir / 'results.json'
    forward_numpy_dir = forward_dir / 'numpy'
    backward_numpy_dir = backward_dir / 'numpy'

    nusc = NuScenes(version='v1.0-trainval',
                    dataroot='data/nuscenes', verbose=True)
    scenes = [scene for scene in nusc.scene if scene['name'] in splits.val]
    config_factory('tracking_nips_2019')
    with open(forward_json, 'r') as f:
        forward_obj_dict = json.load(f)['results']
    if merge:
        with open(backward_json, 'r') as f:
            backward_obj_dict = json.load(f)['results']
    results = {}
    for scene in scenes:
        cur_sample_token = scene['first_sample_token']
        samples = []
        while cur_sample_token != '':
            cur_sample = nusc.get('sample', cur_sample_token)
            samples.append(cur_sample)
            cur_sample_token = cur_sample['next']
        sample_tokens = [sample['token'] for sample in samples]
        results.update({k: [] for k in sample_tokens})
        forward_scene_objs = [forward_obj_dict[sample_token] for sample_token in sample_tokens]
        timestamps = [sample['timestamp'] for sample in samples]
        trajectories = load_trajectories(forward_scene_objs, forward_numpy_dir / scene['name'], timestamps)
        if merge:
            backward_scene_objs = [backward_obj_dict[sample_token] for sample_token in sample_tokens]
            backward_trajectories = load_trajectories(backward_scene_objs, backward_numpy_dir / scene['name'], timestamps)
            trajectories = merge_forward_backward_trajectories(
                trajectories, backward_trajectories,
                visualize_contradictions=False
            )
        # todo
        for _, objs in trajectories.values():
            for obj in objs:
                results[obj.sample_token].append(obj.serialize())
    with open(out_json, 'w') as f:
        json.dump({"meta": {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        }, 'results': results}, f)

if __name__ == '__main__':
    main()