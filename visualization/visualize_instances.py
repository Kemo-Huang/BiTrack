from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import tqdm

from tracking.detections.kitti_detections import get_corr_2d_inds_hungarian
from utils import (Calibration, boxes_to_corners_3d, get_lidar_boxes_from_objs,
                   get_objects_from_label, sigmoid)


def vis_seq(seq_inst_dir: Path, seq_det3d_dir: Path, seq_lidar_dir: Path, calib: Calibration):
    pbar = tqdm.tqdm(list(seq_inst_dir.iterdir()))
    pbar.set_description(seq_inst_dir.name)
    for instance_file in pbar:
        frame = instance_file.stem
        if not (seq_det3d_dir / f'{frame}.txt').exists():
            continue
        
        instance_img =  cv2.imread(str(instance_file), cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(str(img_dir / seq_inst_dir.name / instance_file.name))

        # 3D
        all_instance_ids = np.unique(instance_img)[1:]
        objs = get_objects_from_label(seq_det3d_dir / f'{frame}.txt')
        objs = [obj for obj in objs if obj.cls_type in ('Car', 'Van')]
        objs = [obj for obj in objs if sigmoid(obj.tracking_score) >= 0.1]
        boxes = get_lidar_boxes_from_objs(objs, calib)

        inside_points = [np.fromfile(
        seq_lidar_dir / frame / f'{i}.bin', np.float32).reshape(-1, 4)[:, :3] for i in range(len(boxes))]
        img_hw = instance_img.shape
        corr_2d_inds, similarity = get_corr_2d_inds_hungarian(
            calib, img_hw, inside_points, 'point', instance_img, 'mask', 1, 0.1, all_instance_ids
        )
        valid_mask_3d = corr_2d_inds >= 0
        if np.sum(valid_mask_3d) == 0:
            continue
        valid_2d_ids = all_instance_ids[corr_2d_inds[valid_mask_3d]]
        valid_box_colors = cv2.applyColorMap(id_map[valid_2d_ids], cv2.COLORMAP_RAINBOW).reshape(-1, 3)
        box_colors = np.ones((len(boxes), 3), dtype=np.uint8) * 255
        box_colors[valid_mask_3d] = valid_box_colors

        instance_color_img = id_map[instance_img]
        instance_color_img = cv2.applyColorMap(instance_color_img, cv2.COLORMAP_RAINBOW)
        valid_mask_2d = np.isin(instance_img, valid_2d_ids)
        img[valid_mask_2d] = alpha * instance_color_img[valid_mask_2d] + (1- alpha) * img[valid_mask_2d]
        invalid_mask_2d = (instance_img > 0) & ~valid_mask_2d
        img[invalid_mask_2d] = alpha * np.ones_like(img)[invalid_mask_2d] * 255 + (1- alpha) * img[invalid_mask_2d]

        n = len(boxes)
        corners = boxes_to_corners_3d(boxes).reshape(n*8, 3)
        pts_img, _ = calib.lidar_to_img(corners)
        pts_img = pts_img.reshape(n, 8, 2).astype(int)

        for pts, box_color in zip(pts_img, box_colors):
            if np.sum((pts[:, 0] >= 0) & (pts[:, 1] >= 0) & (pts[:, 0] < img_hw[1]) & (pts[:, 1] < img_hw[0])) < 3:
                continue
            box_color = box_color.tolist()
            cv2.line(img, pts[0], pts[1], color=box_color, thickness=2)
            cv2.line(img, pts[1], pts[2], color=box_color, thickness=2)
            cv2.line(img, pts[2], pts[3], color=box_color, thickness=2)
            cv2.line(img, pts[3], pts[0], color=box_color, thickness=2)
            cv2.line(img, pts[4], pts[5], color=box_color, thickness=2)
            cv2.line(img, pts[5], pts[6], color=box_color, thickness=2)
            cv2.line(img, pts[6], pts[7], color=box_color, thickness=2)
            cv2.line(img, pts[7], pts[4], color=box_color, thickness=2)
            cv2.line(img, pts[0], pts[4], color=box_color, thickness=2)
            cv2.line(img, pts[1], pts[5], color=box_color, thickness=2)
            cv2.line(img, pts[2], pts[6], color=box_color, thickness=2)
            cv2.line(img, pts[3], pts[7], color=box_color, thickness=2)

        if args.save:
            cv2.imwrite(str(vis_out_dir / instance_file.name), img)
        else:
            cv2.imshow(f'{instance_file.stem}', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    instance_dir = Path('data/kitti/tracking/training/seg_out/spatial_embeddings')
    det3d_dir = Path('data/kitti/tracking/training/det3d_out/virconv')
    img_dir = Path('data/kitti/tracking/training/image_02')
    calib_dir = Path('data/kitti/tracking/training/calib')
    lidar_dir = Path('data/kitti/tracking/training/cropped_points/virconv')
    alpha = 0.6
    np.random.seed(1234)
    id_map = np.random.permutation(256).astype(np.uint8)
    parser = ArgumentParser()
    parser.add_argument('--seq', type=str)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    seq = args.seq
    if seq:
        seq = seq.zfill(4)
        vis_out_dir = Path(f'output/kitti/vis_out/instance/{seq}')
        vis_out_dir.mkdir(exist_ok=True, parents=True)
        calib = Calibration(calib_dir / f'{seq}.txt')
        vis_seq(instance_dir / seq, det3d_dir / seq, lidar_dir / seq, calib)
    else:
        for seq_inst_dir in instance_dir.iterdir():
            seq = seq_inst_dir.name
            calib = Calibration(calib_dir / f'{seq}.txt')
            vis_seq(seq_inst_dir, det3d_dir / seq, lidar_dir / seq, calib)
    if args.save:
        print(f'vis results saved in {vis_out_dir}')