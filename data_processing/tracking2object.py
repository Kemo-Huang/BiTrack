import shutil
from argparse import ArgumentParser
from pathlib import Path

import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument("trk_dir", type=str)
    parser.add_argument("obj_dir", type=str)
    args = parser.parse_args()
    trk_dir = Path(args.trk_dir)
    obj_dir = Path(args.obj_dir)
    obj_dir.mkdir(exist_ok=True, parents=True)

    with open("data/kitti/detection/training/seq2sample.txt") as f:
        lines = f.readlines()
    split_lines = [line.split() for line in lines]
    seq2samples = {s[0]: s[1:] for s in split_lines}

    for seq, samples in tqdm.tqdm(seq2samples.items()):
        for i, file in enumerate(sorted((trk_dir / seq).iterdir())):
            shutil.copyfile(file, obj_dir / f"{samples[i]}.txt")


if __name__ == "__main__":
    main()
