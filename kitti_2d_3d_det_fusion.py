import json
import pickle
from argparse import ArgumentParser
from configparser import ConfigParser
from pathlib import Path

import tqdm

from tracking.detections.kitti_detections import get_detection_data
from utils import Calibration, get_poses_from_file, read_seqmap_file


def main():
    parser = ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("split", type=str)
    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config)

    root_dir = Path(config["data"]["root_dir"]) / args.split

    calib_dir = root_dir / "calib"
    oxts_dir = root_dir / "oxts"
    img_hw_dict = json.load(open(root_dir / "img_hw.json"))

    detection_cfg = config["detection"]

    det3d_name = detection_cfg["det3d_name"]
    det3d_dir = root_dir / "det3d_out" / det3d_name
    crop_dir = root_dir / "cropped_points" / det3d_name
    det2d_dir = root_dir / "det2d_out" / detection_cfg["det2d_name"]
    det2d_emb_dir = root_dir / "det2d_emb_out" / detection_cfg["det2d_emb_name"]
    seg_out_dir = root_dir / "seg_out" / detection_cfg["seg_name"]
    seg_emb_dir = root_dir / "seg_emb_out" / detection_cfg["seg_emb_name"]
    det3d_save_name = detection_cfg["det3d_save_name"]
    det3d_save_dir = root_dir / "det3d_out" / det3d_save_name

    use_lidar = detection_cfg.getboolean("use_lidar")
    use_inst = detection_cfg.getboolean("use_inst")
    use_det2d = detection_cfg.getboolean("use_det2d")
    assert not (use_inst and use_det2d)
    use_embed = detection_cfg.getboolean("use_embed")

    if use_inst:
        emb_dir = seg_emb_dir
    elif use_det2d:
        emb_dir = det2d_emb_dir
    else:
        use_embed = False

    seqmap_file = root_dir / f"evaluate_tracking.seqmap.{args.split}"
    frame_num_dict = read_seqmap_file(seqmap_file)

    for seq in frame_num_dict:
        # (seq 0001: missing 177 178 179 180)
        seq_det3d_dir = det3d_dir / seq
        frames = [f.stem for f in seq_det3d_dir.iterdir()]
        num_frames = len(frames)
        seq_det3d_save_dir: Path = det3d_save_dir / seq
        seq_det3d_save_dir.mkdir(parents=True, exist_ok=True)

        if detection_cfg.getboolean("use_pose"):
            # Reads imu data and converts to poses
            poses = get_poses_from_file(oxts_dir / f"{seq}.txt")
        else:
            poses = [None for _ in range(num_frames)]

        # seq calibration data
        calib = Calibration(calib_dir / f"{seq}.txt")

        seq_img_hw_dict = img_hw_dict[seq]

        good_dets_3d = {}
        bad_dets_3d = {}

        pbar = tqdm.tqdm(range(num_frames))
        pbar.set_description(seq)
        for idx in pbar:
            # Retrieves the current frame info
            frame = frames[idx]

            # Gets detections from files
            cur_good_dets, cur_bad_dets_3d, _ = get_detection_data(
                det3d_file=seq_det3d_dir / f"{frame}.txt",
                calib=calib,
                pose=poses[idx],
                img_hw=seq_img_hw_dict[frame],
                lidar_dir=crop_dir / seq / frame if use_lidar else None,
                det2d_file=det2d_dir / seq / f"{frame}.txt" if use_det2d else None,
                seg_file=seg_out_dir / seq / f"{frame}.png" if use_inst else None,
                embed_dir=emb_dir / seq / frame if use_embed else None,
                min_corr_pts=detection_cfg.getfloat("min_corr_pts"),
                min_corr_iou=detection_cfg.getfloat("min_corr_iou"),
                raw_score=detection_cfg.getboolean("raw_score"),
                score_thresh=detection_cfg.getfloat("score_thresh"),
                recover_score_thresh=detection_cfg.getfloat("recover_score_thresh"),
            )

            with open(seq_det3d_save_dir / f"{frame}.txt", "w") as f:
                lines = []
                if cur_good_dets is not None:
                    lines = [trk.to_obj().serialize() for trk in cur_good_dets.objs]
                f.writelines(lines)

            good_dets_3d[frame] = cur_good_dets
            bad_dets_3d[frame] = cur_bad_dets_3d

        with open(seq_det3d_save_dir / "good_dets_3d.pkl", "wb") as f:
            pickle.dump(good_dets_3d, f)

        with open(seq_det3d_save_dir / "bad_dets_3d.pkl", "wb") as f:
            pickle.dump(bad_dets_3d, f)


if __name__ == "__main__":
    main()
