from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def crop_images(label_dir: Path, img_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for seq in range(21):
        seq = str(seq).zfill(4)
        seq_label_dir = label_dir / seq
        pbar = tqdm(list(seq_label_dir.iterdir()))
        pbar.set_description(seq)
        for label_file in pbar:
            frame = label_file.stem
            image = Image.open(img_dir / seq / f"{frame}.png")
            cur_out_dir = out_dir / seq / frame

            seg_image = np.array(Image.open(label_file))
            inst_ids = np.unique(seg_image)[1:]
            for idx, inst_id in enumerate(inst_ids):
                cur_out_dir.mkdir(exist_ok=True, parents=True)
                row_inds, col_inds = np.nonzero(seg_image == inst_id)
                box2d = [
                    np.min(col_inds),
                    np.min(row_inds),
                    np.max(col_inds),
                    np.max(row_inds),
                ]
                cropped_img = image.crop(box2d)
                cropped_img.save(cur_out_dir / f"{idx}.png")


def main():
    img_dir = Path("data/kitti/tracking/training/image_02")
    out_dir = Path("data/kitti/tracking/training/seg_emb_out/spatial_embeddings_img")
    label_dir = Path("data/kitti/tracking/training/seg_out/spatial_embeddings")
    crop_images(label_dir, img_dir, out_dir)


if __name__ == "__main__":
    main()
