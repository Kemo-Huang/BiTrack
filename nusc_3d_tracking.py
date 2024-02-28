import json
import time
from argparse import ArgumentParser
from collections import defaultdict
from configparser import ConfigParser
from pathlib import Path

import numpy as np
import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits

from tracking.detections import Detections
from tracking.tracker import Tracker
from utils import NuscenesObject, visualize_trajectories

CLASSES = [
    "barrier",
    "traffic_cone",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "car",
    "bus",
    "construction_vehicle",
    "trailer",
    "truck",
]


def load_args_and_config():
    parser = ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("tag", type=str)
    parser.add_argument("split", type=str)
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config)
    return args, config


def create_tracker(config: ConfigParser):
    tracking_cfg = config["tracking"]
    tracker = Tracker(
        t_miss=tracking_cfg.getint("t_miss"),
        t_miss_new=tracking_cfg.getint("t_miss_new"),
        t_hit=tracking_cfg.getint("t_hit"),
        match_algorithm=tracking_cfg["match_algorithm"],
        aff_thresh=tracking_cfg.getfloat("dis_thresh"),
        ang_thresh=tracking_cfg.getfloat("ang_thresh"),
        app_thresh=tracking_cfg.getfloat("app_thresh"),
        ent_ex_score=tracking_cfg.getfloat("ent_ex_score"),
        app_m=tracking_cfg.getfloat("app_m"),
        offline=tracking_cfg.getboolean("offline"),
        p=tracking_cfg.getfloat("p"),
        q=tracking_cfg.getfloat("q"),
        ang_vel=tracking_cfg.getboolean("ang_vel"),
        vel_reinit=tracking_cfg.getboolean("vel_reinit"),
        sim_metric=tracking_cfg["sim_metric"],
    )
    return tracker


def get_num_passed_frames(frame, last_frame, backward):
    if last_frame is None:
        last_frame = int(frame)
        num_passed_frames = 1
    else:
        if backward:
            num_passed_frames = last_frame - int(frame)
            last_frame -= num_passed_frames
        else:
            num_passed_frames = int(frame) - last_frame
            last_frame += num_passed_frames
    return last_frame, num_passed_frames


def main():
    args, config = load_args_and_config()

    tracker = create_tracker(config)

    visualization_cfg = config["visualization"]
    backward = args.backward

    tracking_out_dir = Path(f"output/nuscenes/{args.split}/{args.tag}")
    tracking_out_dir.mkdir(parents=True, exist_ok=True)

    with open(tracking_out_dir / "config.ini", "w") as f:
        config.write(f)

    root = config["data"]["root_dir"]
    nusc = NuScenes(version="v1.0-test" if "test" in args.split else "v1.0-trainval", dataroot=root, verbose=False)
    scene_names = splits.test if "test" in args.split else splits.val
    scenes = [x for x in nusc.scene if x["name"] in scene_names]
    with open(Path(root) / "results_nusc_val.json", "r") as f:
        det_results = json.load(f)
    meta = det_results["meta"]
    det_results = det_results["results"]
    trk_results = []
    det_results_in_scenes = {}
    for scene in scenes:
        print("==============================")
        num_frames = scene["nbr_samples"]
        for det_class in CLASSES:
            scene_dets = []
            sample_token = scene["first_sample_token"]
            while sample_token:
                sample_dict = nusc.get("sample", sample_token)
                sample_dets = det_results[sample_token]
                det_results_in_scenes[sample_token] = sample_dets
                sample_dets = [
                    x for x in sample_dets if "detection_name" in x and x["detection_name"] == det_class
                ]
                sample_dets = [NuscenesObject(x) for x in sample_dets]
                dets = Detections(
                    np.array([x.to_box() for x in sample_dets]), sample_dets, None, None, None
                )
                scene_dets.append(dets)
                sample_token = sample_dict["next"]

            # init
            tracker.reset()
            offline_trajectories = {}
            time_cost = 0

            pbar = tqdm.tqdm(
                list(reversed(range(num_frames))) if backward else range(num_frames)
            )
            pbar.set_description(f'{scene["name"]} {det_class}'.ljust(35))

            for idx in pbar:
                cur_good_dets = scene_dets[idx]
                # Perfroms tracking for the current frame
                start_time = time.time()
                _, pred_boxes = tracker.predict()
                matched, entry_dets, exit_trks, false_trks = tracker.associate(
                    pred_boxes, cur_good_dets
                )
                tracker.update(
                    matched, entry_dets, exit_trks, false_trks, cur_good_dets
                )
                online_trks, dead_tracks = tracker.track_management()
                end_time = time.time()
                time_cost += end_time - start_time

                # Saves data to strings
                if tracker.offline:
                    for trk in dead_tracks:
                        if trk.max_hits >= tracker.t_hit:
                            for obj in trk.objs:
                                obj.tracking_id = trk.id
                            offline_trajectories[trk.id] = (
                                [trk.boxes[::-1], trk.objs[::-1]]
                                if backward
                                else [trk.boxes, trk.objs]
                            )

                else:
                    for trk in online_trks:
                        trk.obj.tracking_id = trk.id
                    trk_results.extend([trk.obj.serialize() for trk in online_trks])

            if tracker.offline:
                for trk in tracker.tracks:
                    if trk.max_hits >= tracker.t_hit:
                        for obj in trk.objs:
                            obj.tracking_id = trk.id
                        offline_trajectories[trk.id] = (
                            [trk.boxes[::-1], trk.objs[::-1]]
                            if backward
                            else [trk.boxes, trk.objs]
                        )

                if visualization_cfg.getboolean("trajectory"):
                    visualize_trajectories(
                        trajectories=[
                            boxes for boxes, _ in offline_trajectories.values()
                        ],
                    )
                for _, objs in offline_trajectories.values():
                    trk_results.extend([obj.serialize() for obj in objs])

    trk_results_dict = defaultdict(list)
    for obj in trk_results:
        trk_results_dict[obj["sample_token"]].append(obj)
    trk_results = {"meta": meta, "results": trk_results_dict}
    with open(tracking_out_dir / "results.json", "w") as f:
        json.dump(trk_results, f)
    with open("data/nuscenes/mini_val_results.json", "w") as f:
        json.dump({"meta": meta, "results": det_results_in_scenes}, f)

if __name__ == "__main__":
    main()
