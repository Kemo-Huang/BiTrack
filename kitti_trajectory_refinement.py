import json
import shutil
from argparse import ArgumentParser
from configparser import ConfigParser
from pathlib import Path

import tqdm

from tracking.trajectory_clustering_split_and_recombination import \
    merge_forward_backward_trajectories
from tracking.trajectory_completion import linear_interpolation
from tracking.trajectory_refinement import (box_size_weighted_mean,
                                            gaussian_smoothing)
from utils import (Calibration, read_kitti_trajectories_from_file,
                   read_seqmap_file, write_kitti_trajectories_to_file)


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

    out_dir = Path(f"output/kitti/{args.split}/{args.tag}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_data_dir = out_dir / "data"
    out_data_dir.mkdir(parents=True)
    forward_dir = Path(f"output/kitti/{args.split}/{args.forward_tag}")
    backward_dir = Path(f"output/kitti/{args.split}/{args.backward_tag}")
    forward_data_dir = forward_dir / "data"
    backward_data_dir = backward_dir / "data"

    root_dir = Path(config["data"]["root_dir"])
    split_dir = root_dir / ("testing" if "test" in args.split else "training")
    calib_dir = split_dir / "calib"
    img_hw_dict = json.load(open(split_dir / "img_hw.json"))

    seqmap_file = split_dir / f"evaluate_tracking.seqmap.{args.split}"
    frame_num_dict = read_seqmap_file(seqmap_file)
    raw_score = config["detection"].getboolean("raw_score")

    for seq in tqdm.tqdm(frame_num_dict):
        calib = Calibration(calib_dir / f"{seq}.txt")
        cur_img_hw_dict = img_hw_dict[seq]
        trajectories = read_kitti_trajectories_from_file(
            seq, forward_data_dir, calib, cur_img_hw_dict, raw_score=raw_score
        )
        if refinement_cfg.getboolean("merge"):
            backward_trajectories = read_kitti_trajectories_from_file(
                seq, backward_data_dir, calib, cur_img_hw_dict, raw_score=raw_score
            )
            trajectories = merge_forward_backward_trajectories(
                trajectories,
                backward_trajectories,
                visualize_contradictions=visualization_cfg.getboolean("contradiction"),
            )
        if refinement_cfg.getboolean("box_size_fusion"):
            trajectories = box_size_weighted_mean(
                trajectories,
                calib,
                cur_img_hw_dict,
                exponent=refinement_cfg.getfloat("exponent"),
            )
        if refinement_cfg.getboolean("interp"):
            trajectories = linear_interpolation(
                trajectories,
                calib,
                img_hw_dict=cur_img_hw_dict,
                interp_max_interval=refinement_cfg.getint("interp_max_interval"),
                score_thresh=refinement_cfg.getfloat("score_thresh"),
                ignore_thresh=refinement_cfg.getfloat("ignore_thresh"),
                nms_thresh=refinement_cfg.getfloat("nms_thresh"),
                visualize=visualization_cfg.getboolean("interpolation"),
            )
        if refinement_cfg.getboolean("smooth"):
            trajectories = gaussian_smoothing(
                trajectories, calib, cur_img_hw_dict, refinement_cfg.getfloat("tau")
            )
        write_kitti_trajectories_to_file(seq, trajectories, out_data_dir)


if __name__ == "__main__":
    main()
