import json
from argparse import ArgumentParser
from configparser import ConfigParser
from pathlib import Path

import tqdm
from PIL import Image


def save_img_hw_json(root_dir: Path):
    img_hw_dict = {}
    img_dir = root_dir / "image_02"
    for seq_dir in img_dir.iterdir():
        cur_seq_img_hw_dict = {}
        pbar = tqdm.tqdm(list(seq_dir.iterdir()))
        pbar.set_description(seq_dir.name)
        for img_path in pbar:
            img = Image.open(img_path)
            cur_seq_img_hw_dict[img_path.stem] = (img.height, img.width)
        img_hw_dict[seq_dir.name] = cur_seq_img_hw_dict

    with open(root_dir / "img_hw.json", "w") as f:
        json.dump(img_hw_dict, f)


def main():
    parser = ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("split", type=str)
    args = parser.parse_args()
    config = ConfigParser()
    config.read(args.config)
    root_dir = Path(config["data"]["root_dir"]) / args.split
    save_img_hw_json(root_dir)


if __name__ == "__main__":
    main()
