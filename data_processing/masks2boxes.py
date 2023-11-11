from pathlib import Path

import numpy as np
import tqdm
from PIL import Image


def convert_gt_instances_to_imgs():
    instance_dir = Path("data/kitti/tracking/training/instances")
    seg_out_dir = Path("data/kitti/tracking/training/seg_out/gt")
    seg_out_dir.mkdir(exist_ok=True, parents=True)
    for seq in range(21):
        seq = str(seq).zfill(4)
        seq_dir = instance_dir / seq
        seq_out_dir = seg_out_dir / seq
        seq_out_dir.mkdir(exist_ok=True)
        bar = tqdm.tqdm(list(seq_dir.iterdir()))
        bar.set_description(seq)
        for img_file in bar:
            img = np.asarray(Image.open(img_file))  # uin16
            new_img = np.zeros(img.shape[:2], dtype=np.uint8)
            obj_ids = np.unique(img)
            for obj_id in obj_ids:
                class_id = obj_id // 1000
                if class_id == 1:
                    inst_id = obj_id % 1000 + 1
                    assert inst_id <= 255
                    new_img[img == obj_id] = inst_id
            Image.fromarray(new_img).save(seq_out_dir / img_file.name)


def convert_to_boxes():
    seg_out_dir = Path("data/kitti/tracking/training/seg_out/spatial_embeddings")
    det_out_dir = Path("data/kitti/tracking/training/det2d_out/spatial_embeddings")
    det_out_dir.mkdir(exist_ok=True, parents=True)
    for seq in range(21):
        seq = str(seq).zfill(4)
        seq_dir = seg_out_dir / seq
        seq_out_dir = det_out_dir / seq
        seq_out_dir.mkdir(exist_ok=True)
        bar = tqdm.tqdm(list(seq_dir.iterdir()))
        bar.set_description(seq)
        for img_file in bar:
            img = np.asarray(Image.open(img_file))
            obj_ids = np.unique(img)[1:]
            lines = []
            for obj_id in obj_ids:
                row_inds, col_inds = np.nonzero(img == obj_id)
                box2d = [
                    np.min(col_inds),
                    np.min(row_inds),
                    np.max(col_inds),
                    np.max(row_inds),
                ]
                lines.append(" ".join([str(x) for x in box2d]) + f" 1.0 {obj_id}\n")
            with open(seq_out_dir / f"{img_file.stem}.txt", "w") as f:
                f.writelines(lines)


if __name__ == "__main__":
    convert_to_boxes()
