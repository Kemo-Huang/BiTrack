from pathlib import Path

import numpy as np
import tqdm
from PIL import Image


def main():
    instance_dir = Path('data/tracking/seg_instances')
    vis_dir = Path('data/tracking/inst_vis')
    vis_dir.mkdir(exist_ok=True)
    img_dir = Path('data/tracking/image_02')

    for seq in instance_dir.iterdir():
        pbar = tqdm.tqdm(list(seq.iterdir()))
        pbar.set_description(seq.name)
        for frame in pbar:
            instance_img = np.asarray(Image.open(frame))
            img = np.array(Image.open(img_dir / seq.name / frame.name))
            for inst_id in np.unique(instance_img)[1:]:
                inst_mask = instance_img == inst_id
                row_ind, col_ind = np.nonzero(inst_mask)
                inst_img = np.ones_like(img) * 255  # white background

                inst_img[inst_mask] = img[inst_mask]

                y0 = np.min(row_ind)
                y1 = np.max(row_ind)
                x0 = np.min(col_ind)
                x1 = np.max(col_ind)
                inst_img = Image.fromarray(inst_img)
                inst_img = inst_img.crop((x0, y0, x1, y1))
                cur_out_dir = vis_dir / seq.name/ frame.stem
                cur_out_dir.mkdir(exist_ok=True, parents=True)
                inst_img.save(cur_out_dir / f'{inst_id}.png')

if __name__ == '__main__':
    main()