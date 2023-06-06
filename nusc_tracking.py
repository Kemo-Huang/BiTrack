
import json
import shutil
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from nuscenes.eval.common.config import config_factory
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from tqdm import tqdm

from tracking.detections.nusc_detections import (get_detection_data,
                                                 get_detection_data_all)
from tracking.tracker import Tracker
from utils import visualize_trajectories, write_nusc_trajectories_to_file


def scan(backward=False):
    results = {}
    for scene in scenes:
        cur_sample_token = scene['first_sample_token']
        samples = []
        while cur_sample_token != '':
            cur_sample = nusc.get('sample', cur_sample_token)
            samples.append(cur_sample)
            cur_sample_token = cur_sample['next']
        # init
        tracker.reset()
        offline_trajectories = {}
        bad_dets_3d = []
        bad_trks = []
        time_cost = 0
        # Reads imu data and converts to poses
        pbar = tqdm(list(reversed(range(len(samples))))
                    if backward else range(len(samples)))
        for idx in pbar:
            pbar.set_description(scene['name'])
            sample_token = samples[idx]['token']
            good_dets = get_detection_data(objs_dict, sample_token)

            # Perfroms tracking for the current frame
            start_time = time.time()
            pred_ids, pred_boxes = tracker.predict(1)
            matched, entry_dets, exit_trks, false_trks = tracker.associate(
                pred_boxes, good_dets)
            tracker.update(matched, entry_dets, exit_trks,
                           false_trks, good_dets)
            online_trks, dead_tracks = tracker.track_management()
            end_time = time.time()
            time_cost += end_time - start_time

            if offline:
                for trk in dead_tracks:
                    if trk.max_hits >= tracker.t_hit:
                        for obj in trk.objs:
                            obj.tracking_id = trk.id
                        offline_trajectories[trk.id] = [
                            trk.boxes[::-1], trk.objs[::-1]] if backward else [trk.boxes, trk.objs]
                    else:
                        bad_trks.append(trk)
                results[sample_token] = []  # init
            else:
                for trk in online_trks:
                    trk.obj.tracking_id = trk.id
                results[sample_token] = [trk.obj.serialize()
                                         for trk in online_trks]

        if offline:
            for trk in tracker.tracks:
                if trk.max_hits >= tracker.t_hit:
                    for obj in trk.objs:
                        obj.tracking_id = trk.id
                    offline_trajectories[trk.id] = [
                        trk.boxes[::-1], trk.objs[::-1]] if backward else [trk.boxes, trk.objs]
                else:
                    bad_trks.append(trk)

            # print('bad tracks', sum([len(trk.boxes) for trk in bad_trks]))
            # print('bad det 3d', sum(
            #     [len(dets.boxes) if dets is not None else 0 for _, dets in bad_dets_3d]))
            # print('bad det 2d', sum(
            #     [len(dets.inst_ids) if dets is not None else 0 for _, dets in bad_dets_2d]))

            if visualize:
                visualize_trajectories(
                    trajectories=[boxes for boxes,
                                  _ in offline_trajectories.values()],
                    other_boxes=None if use_gt or not visualize_noise else np.concatenate(
                        [dets.boxes for _, dets in bad_dets_3d if dets is not None])
                )
            cur_scene_results = write_nusc_trajectories_to_file(
                scene, offline_trajectories, tracking_out_numpy_dir)
            results.update(cur_scene_results)

    with open(tracking_out_json, 'w') as f:
        json.dump({"meta": {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        }, 'results': results}, f)


if __name__ == '__main__':
    # ============================================================================
    # Configs
    det_name = 'centerpoint_output'
    offline = True
    score_thresh = 0
    use_pose = False
    visualize = False
    visualize_noise = False
    use_gt = False
    use_lidar = True
    use_inst = True
    use_embed = True
    det_min_pc_iou = 0
    det_min_n_pts = 1
    pred_pc_iou_thresh = 0.5
    pred_min_n_pts = 5
    backward_scan = True

    tracker = Tracker(
        t_miss=10,
        t_hit=3,
        match_algorithm=0,
        score_thresh=0,
        aff_thresh=0.5,
        app_thresh=0.5,
        ent_ex_score=0.6,
        offline=offline,
        filter='kf'
    )
    # ============================================================================

    parser = ArgumentParser()
    parser.add_argument('tag', type=str)
    args = parser.parse_args()
    tracking_out_dir = Path(f'output/nusc/{args.tag}')
    tracking_out_json = tracking_out_dir / 'results.json'
    tracking_out_numpy_dir = tracking_out_dir / 'numpy'
    tracking_out_numpy_dir.mkdir(parents=True, exist_ok=True)
    for numpy_dir in tracking_out_numpy_dir.iterdir():
        shutil.rmtree(numpy_dir)
    if use_gt:
        use_lidar = False
        use_inst = False

    config_factory('tracking_nips_2019')
    objs_dict = get_detection_data_all(f'data/nuscenes/{det_name}.json')
    nusc = NuScenes(version='v1.0-trainval',
                    dataroot='data/nuscenes', verbose=True)
    scenes = [scene for scene in nusc.scene if scene['name'] in splits.val]
    scan(backward_scan)