import shutil
import sys
from pathlib import Path


def main():
    dir_name = sys.argv[1]
    obj_dir = Path(f"data/kitti/detection/{dir_name}")
    trk_dir = Path(f"data/kitti/tracking/training/det2d_out/{dir_name}")
    trk_dir.mkdir(exist_ok=True, parents=True)

    with open("data/kitti/detection/sample2seq.txt") as f:
        lines = f.readlines()
    split_lines = [line.split() for line in lines]
    sample2seq = {s[0]: s[1:] for s in split_lines}

    for sample_file in obj_dir.iterdir():
        seq, frame = sample2seq[sample_file.stem]
        cur_dir = trk_dir / seq
        cur_dir.mkdir(exist_ok=True)
        shutil.copyfile(sample_file, cur_dir / f"{frame}.txt")


if __name__ == "__main__":
    main()
