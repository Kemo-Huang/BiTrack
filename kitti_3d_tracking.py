import pickle
import shutil
import time
from argparse import ArgumentParser
from configparser import ConfigParser
from pathlib import Path

import numpy as np
import tqdm

from tracking.tracker import Tracker
from utils import (read_seqmap_file, visualize_trajectories,
                   write_kitti_trajectories_to_file)


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

    root_dir = Path(config["data"]["root_dir"])
    split_dir = root_dir / ("testing" if "test" in args.split else "training")
    detection_cfg = config["detection"]

    det3d_name = detection_cfg["det3d_name"]
    det3d_dir = split_dir / "det3d_out" / det3d_name
    det3d_save_name = detection_cfg["det3d_save_name"]
    det3d_save_dir = split_dir / "det3d_out" / det3d_save_name

    tracker = create_tracker(config)

    visualization_cfg = config["visualization"]
    backward = args.backward

    seqmap_file = split_dir / f"evaluate_tracking.seqmap.{args.split}"
    frame_num_dict = read_seqmap_file(seqmap_file)

    tracking_out_dir = Path(f"output/kitti/{args.split}/{args.tag}")
    if tracking_out_dir.exists():
        shutil.rmtree(tracking_out_dir)
    tracking_out_txt_dir = tracking_out_dir / "data"
    tracking_out_txt_dir.mkdir(parents=True)

    with open(tracking_out_dir / "config.ini", "w") as f:
        config.write(f)

    for seq in frame_num_dict:
        # (seq 0001: missing 177 178 179 180)
        seq_det3d_dir = det3d_dir / seq
        frames = [f.stem for f in seq_det3d_dir.iterdir()]
        num_frames = len(frames)
        seq_det3d_save_dir: Path = det3d_save_dir / seq

        # init
        tracker.reset()
        last_frame = None
        cur_seq_output_lines = []
        offline_trajectories = {}
        time_cost = 0

        pbar = tqdm.tqdm(
            list(reversed(range(num_frames))) if backward else range(num_frames)
        )
        pbar.set_description(seq)

        with open(seq_det3d_save_dir / "good_dets_3d.pkl", "rb") as f:
            good_dets_3d = pickle.load(f)

        if visualization_cfg.getboolean("trajectory"):
            with open(seq_det3d_save_dir / "bad_dets_3d.pkl", "rb") as f:
                bad_dets_3d = pickle.load(f)

        for idx in pbar:
            frame = frames[idx]
            last_frame, num_passed_frames = get_num_passed_frames(
                frame, last_frame, backward
            )
            cur_good_dets = good_dets_3d[frame]

            # Perfroms tracking for the current frame
            start_time = time.time()
            _, pred_boxes = tracker.predict(num_passed_frames)
            matched, entry_dets, exit_trks, false_trks = tracker.associate(
                pred_boxes, cur_good_dets
            )
            tracker.update(matched, entry_dets, exit_trks, false_trks, cur_good_dets)
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
                cur_seq_output_lines.extend(
                    [trk.obj.serialize() for trk in online_trks]
                )

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
                    trajectories=[boxes for boxes, _ in offline_trajectories.values()],
                    other_boxes=np.concatenate(
                        [
                            dets.boxes
                            for dets in bad_dets_3d.values()
                            if dets is not None
                        ]
                    )
                    if visualization_cfg.getboolean("det_noise")
                    else None,
                )
            write_kitti_trajectories_to_file(
                seq, offline_trajectories, tracking_out_txt_dir
            )

        else:
            # writes online results
            with open(tracking_out_txt_dir / f"{seq}.txt", "w") as f:
                f.writelines(cur_seq_output_lines)


if __name__ == "__main__":
    main()
