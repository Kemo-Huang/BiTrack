import json
from argparse import ArgumentParser
from collections import defaultdict
from configparser import ConfigParser
from pathlib import Path

import numpy as np
import tqdm
from nuscenes import NuScenes
from nuscenes.utils import splits

from tracking.trajectory_clustering_split_and_recombination import \
    merge_forward_backward_trajectories
from tracking.trajectory_completion import linear_interpolation_nusc
from tracking.trajectory_refinement import (box_size_weighted_mean_nusc,
                                            gaussian_smoothing_nusc)
from utils import NuscenesObject

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


def get_trajectories(trk_results, det_class, sample_token_dict):
    trajectories = defaultdict(list)
    all_results = sum(trk_results.values(), [])
    for x in all_results:
        if x["sample_token"] in sample_token_dict and x["tracking_name"] == det_class:
            obj = NuscenesObject(x)
            obj.sample_id = sample_token_dict[x["sample_token"]]
            obj.data["sample_token"] = obj.sample_id
            trajectories[x["tracking_id"]].append(obj)
    for k, v in trajectories.items():
        v = sorted(v, key=lambda x: x.data["sample_token"])
        trajectories[k] = (np.array([x.to_box() for x in v]).tolist(), v)
    return trajectories


def main():
    parser = ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("tag", type=str)
    parser.add_argument("split", type=str)
    parser.add_argument("forward_tag", type=str)
    parser.add_argument("backward_tag", type=str)
    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config)
    refinement_cfg = config["refinement"]
    visualization_cfg = config["visualization"]

    tracking_out_dir = Path(f"output/nuscenes/{args.split}/{args.tag}")
    tracking_out_dir.mkdir(exist_ok=True, parents=True)
    forward_dir = Path(f"output/nuscenes/{args.split}/{args.forward_tag}")
    backward_dir = Path(f"output/nuscenes/{args.split}/{args.backward_tag}")

    root = Path(config["data"]["root_dir"])

    nusc = NuScenes(version="v1.0-test" if "test" in args.split else "v1.0-trainval", dataroot=root, verbose=False)
    scene_names = splits.test if "test" in args.split else splits.val
    scenes = [x for x in nusc.scene if x["name"] in scene_names]
    with open(forward_dir / "results.json", "r") as f:
        forward_dets = json.load(f)
    meta = forward_dets["meta"]
    forward_dets = forward_dets["results"]

    if refinement_cfg.getboolean("merge"):
        with open(backward_dir / "results.json", "r") as f:
            backward_dets = json.load(f)
        backward_dets = backward_dets["results"]

    trk_results = []
    for scene in tqdm.tqdm(scenes):
        sample_token = scene["first_sample_token"]
        sample_token_dict = dict()
        cnt = 0
        sample_tokens = []
        while sample_token:
            sample_tokens.append(sample_token)
            sample_token_dict[sample_token] = cnt
            cnt += 1
            sample_dict = nusc.get("sample", sample_token)
            sample_token = sample_dict["next"]

        cur_scene_results = []
        for det_class in CLASSES:
            trajectories = get_trajectories(forward_dets, det_class, sample_token_dict)

            if refinement_cfg.getboolean("merge"):
                backward_trajectories = get_trajectories(
                    backward_dets, det_class, sample_token_dict
                )
                trajectories = merge_forward_backward_trajectories(
                    trajectories,
                    backward_trajectories,
                    visualize_contradictions=visualization_cfg.getboolean(
                        "contradiction"
                    ),
                )

            if refinement_cfg.getboolean("box_size_fusion"):
                trajectories = box_size_weighted_mean_nusc(
                    trajectories,
                    exponent=refinement_cfg.getfloat("exponent"),
                )

            if refinement_cfg.getboolean("interp"):
                trajectories = linear_interpolation_nusc(
                    trajectories,
                    interp_max_interval=refinement_cfg.getint("interp_max_interval"),
                    score_thresh=refinement_cfg.getfloat("score_thresh"),
                    ignore_thresh=refinement_cfg.getfloat("ignore_thresh"),
                    nms_thresh=refinement_cfg.getfloat("nms_thresh"),
                    visualize=visualization_cfg.getboolean("interpolation"),
                )

            if refinement_cfg.getboolean("smooth"):
                trajectories = gaussian_smoothing_nusc(
                    trajectories, refinement_cfg.getfloat("tau")
                )

            for _, objs in trajectories.values():
                cur_scene_results.extend([obj.serialize() for obj in objs])

        for x in cur_scene_results:
            x["sample_token"] = sample_tokens[x["sample_token"]]

        trk_results.extend(cur_scene_results)
    trk_results_dict = defaultdict(list)
    for obj in trk_results:
        trk_results_dict[obj["sample_token"]].append(obj)
    with open(tracking_out_dir / "results.json", "w") as f:
        json.dump({"meta": meta, "results": trk_results_dict}, f)


if __name__ == "__main__":
    main()
