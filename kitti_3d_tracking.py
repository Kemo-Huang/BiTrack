import json
import shutil
import time
from argparse import ArgumentParser
from configparser import ConfigParser
from pathlib import Path

import numpy as np
import tqdm

from tracking.detections.kitti_detections import get_detection_data
from tracking.tracker import Tracker
from utils import (Calibration, get_poses_from_file, read_seqmap_file,
                   visualize_trajectories, write_kitti_trajectories_to_file)


def main():
    parser = ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('tag', type=str)
    parser.add_argument('split', type=str)
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config)

    root_dir = Path(config['data']['root_dir'])
    split_dir = root_dir / ('testing' if args.split == 'test' else 'training')
    calib_dir = split_dir / 'calib'
    oxts_dir = split_dir / 'oxts'
    img_hw_dict = json.load(open(split_dir / 'img_hw.json'))

    det3d_name = config['data']['det3d_name']
    det3d_dir = split_dir / 'det3d_out' / det3d_name
    crop_dir = split_dir / 'cropped_points' / det3d_name
    det2d_dir = split_dir / 'det2d_out' / config['data']['det2d_name']
    det2d_emb_dir = split_dir / 'det2d_emb_out' / config['data']['det2d_emb_name']
    seg_out_dir = split_dir / 'seg_out' / config['data']['seg_name']
    seg_emb_dir = split_dir / 'seg_emb_out' / config['data']['seg_emb_name']
    det3d_save_name = config['data']['det3d_save_name']
    det3d_save_dir = split_dir / 'det3d_out' / det3d_save_name

    tracking_out_dir = Path(f'output/kitti/{args.split}/{args.tag}')
    if tracking_out_dir.exists():
        shutil.rmtree(tracking_out_dir)
    tracking_out_txt_dir = tracking_out_dir / 'data'
    tracking_out_txt_dir.mkdir(parents=True)

    tracking_cfg = config['tracking']

    offline = tracking_cfg.getboolean('offline')
    tracker = Tracker(
        t_miss=tracking_cfg.getint('t_miss'),
        t_miss_new=tracking_cfg.getint('t_miss_new'),
        t_hit=tracking_cfg.getint('t_hit'),
        match_algorithm=tracking_cfg['match_algorithm'],
        aff_thresh=tracking_cfg.getfloat('dis_thresh'),
        ang_thresh=tracking_cfg.getfloat('ang_thresh'),
        app_thresh=tracking_cfg.getfloat('app_thresh'),
        ent_ex_score=tracking_cfg.getfloat('ent_ex_score'),
        app_m=tracking_cfg.getfloat('app_m'),
        offline=offline,
        p=tracking_cfg.getfloat('p'),
        q=tracking_cfg.getfloat('q'),
        ang_vel=tracking_cfg.getboolean('ang_vel'),
        vel_reinit=tracking_cfg.getboolean('vel_reinit'),
        sim_metric=tracking_cfg['sim_metric'],
    )

    visualization_cfg = config['visualization']

    use_lidar = tracking_cfg.getboolean('use_lidar')
    use_inst = tracking_cfg.getboolean('use_inst')
    use_det2d = tracking_cfg.getboolean('use_det2d')
    assert not (use_inst and use_det2d)
    use_embed = tracking_cfg.getboolean('use_embed')

    if use_inst:
        emb_dir = seg_emb_dir
    elif use_det2d:
        emb_dir = det2d_emb_dir
    else:
        use_embed = False
    
    backward = args.backward
    
    seqmap_file = split_dir / f'evaluate_tracking.seqmap.{args.split}'
    frame_num_dict = read_seqmap_file(seqmap_file)

    for seq in frame_num_dict:
        # (seq 0001: missing 177 178 179 180)
        seq_det3d_dir = det3d_dir / seq
        frames = [f.stem for f in seq_det3d_dir.iterdir()]
        num_frames = len(frames)
        seq_det3d_save_dir: Path = det3d_save_dir / seq
        seq_det3d_save_dir.mkdir(parents=True, exist_ok=True)
        
        # init
        tracker.reset()
        last_frame = None
        cur_seq_output_lines = []
        offline_trajectories = {}
        bad_dets_2d = []
        bad_dets_3d = []
        bad_preds_3d = []
        bad_trks = []
        time_cost = 0

        if tracking_cfg.getboolean('use_pose'):
            # Reads imu data and converts to poses
            poses = get_poses_from_file(oxts_dir / f'{seq}.txt')
        else:
            poses = [None for _ in range(num_frames)]

        # seq calibration data
        calib = Calibration(calib_dir / f'{seq}.txt')

        seq_img_hw_dict = img_hw_dict[seq]

        pbar = tqdm.tqdm(list(reversed(range(num_frames))) if backward else range(num_frames))
        pbar.set_description(seq)
        for idx in pbar:
            # Retrieves the current frame info
            frame = frames[idx]
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

            # Gets detections from files
            good_dets, cur_bad_dets_3d, cur_bad_dets_2d = get_detection_data(
                det3d_file=seq_det3d_dir / f'{frame}.txt',
                calib=calib,
                pose=poses[idx],
                img_hw=seq_img_hw_dict[frame],
                lidar_dir=crop_dir / seq / frame if use_lidar else None,
                det2d_file=det2d_dir / seq / f'{frame}.txt' if use_det2d else None,
                seg_file=seg_out_dir / seq / f'{frame}.png' if use_inst else None,
                embed_dir=emb_dir / seq / frame if use_embed else None,
                min_corr_pts=tracking_cfg.getfloat('min_corr_pts'),
                min_corr_iou=tracking_cfg.getfloat('min_corr_iou'),
                raw_score=config['data'].getboolean('raw_score'),
                score_thresh=tracking_cfg.getfloat('score_thresh'),
                recover_score_thresh=tracking_cfg.getfloat('recover_score_thresh')
            )

            with open(seq_det3d_save_dir / f'{frame}.txt', 'w') as f:
                lines = []
                if good_dets is not None:
                    lines = [trk.to_obj().serialize() for trk in good_dets.objs]
                f.writelines(lines)

            # Perfroms tracking for the current frame
            start_time = time.time()
            pred_ids, pred_boxes = tracker.predict(num_passed_frames)
            matched, entry_dets, exit_trks, false_trks = tracker.associate(
                pred_boxes, good_dets)
            tracker.update(matched, entry_dets, exit_trks,
                           false_trks, good_dets)
            online_trks, dead_tracks = tracker.track_management()
            end_time = time.time()
            time_cost += end_time - start_time

            bad_dets_3d.append((frame, cur_bad_dets_3d))
            bad_dets_2d.append((frame, cur_bad_dets_2d))
            bad_preds_3d.append(
                (frame, pred_ids[exit_trks], pred_boxes[exit_trks]))

            # Saves data to strings
            if offline:
                for trk in dead_tracks:
                    if trk.max_hits >= tracker.t_hit:
                        for obj in trk.objs:
                            obj.tracking_id = trk.id
                        offline_trajectories[trk.id] = [
                            trk.boxes[::-1], trk.objs[::-1]] if backward else [trk.boxes, trk.objs]
                    else:
                        bad_trks.append(trk)

            else:
                for trk in online_trks:
                    trk.obj.tracking_id = trk.id
                cur_seq_output_lines.extend(
                    [trk.obj.serialize() for trk in online_trks])

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

            if visualization_cfg.getboolean('trajectory'):
                visualize_trajectories(
                    trajectories=[boxes for boxes, _ in offline_trajectories.values()],
                    other_boxes=np.concatenate([dets.boxes for _, dets in bad_dets_3d if dets is not None]) if visualization_cfg.getboolean('det_noise') else None
                )
            write_kitti_trajectories_to_file(seq, offline_trajectories, tracking_out_txt_dir)

        else:
            # writes online results
            with open(tracking_out_txt_dir / f'{seq}.txt', 'w') as f:
                f.writelines(cur_seq_output_lines)



if __name__ == '__main__':
    main()