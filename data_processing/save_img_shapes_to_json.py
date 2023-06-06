import json
from pathlib import Path

import tqdm
from PIL import Image


def save_img_hw_json(split):
    img_hw_dict = {}
    img_dir = Path(f'data/kitti/tracking/{split}/image_02')
    for seq_dir in img_dir.iterdir():
        cur_seq_img_hw_dict = {}
        for img_path in tqdm.tqdm(list(seq_dir.iterdir())):
            img = Image.open(img_path)
            cur_seq_img_hw_dict[img_path.stem] = (img.height, img.width)
        img_hw_dict[seq_dir.name] = cur_seq_img_hw_dict

    with open(f'data/kitti/tracking/{split}/img_hw.json', 'w') as f:
        json.dump(img_hw_dict, f)


def main():
    # save_img_hw_json('training')
    save_img_hw_json('testing')


if __name__ == '__main__':
    main()
