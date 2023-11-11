from pathlib import Path

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
            with open(label_file) as f:
                lines = f.readlines()
            objs = [line.split() for line in lines]

            frame = label_file.stem
            image = Image.open(img_dir / seq / f"{frame}.png")
            cur_out_dir = out_dir / seq / frame
            for idx, obj in enumerate(objs):
                cur_out_dir.mkdir(exist_ok=True, parents=True)
                box2d = list(map(int, map(float, obj[1:5])))
                cropped_img = image.crop(box2d)
                cropped_img.save(cur_out_dir / f"{idx}.png")


def main():
    img_dir = Path("data/kitti/tracking/training/image_02")
    out_dir = Path("data/kitti/tracking/training/det2d_emb_out/yolox")
    label_dir = Path("data/kitti/tracking/training/det2d_out/yolox")
    crop_images(label_dir, img_dir, out_dir)


if __name__ == "__main__":
    main()
