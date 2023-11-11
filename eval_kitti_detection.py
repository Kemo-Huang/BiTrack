from argparse import ArgumentParser
from pathlib import Path

from detection.kitti_object_eval_python.eval import get_official_eval_result
from utils import get_objects_from_label


def get_label_annos(label_dir, sample_ids=None):
    label_dir = Path(label_dir)
    if sample_ids is None:
        label_files = label_dir.glob("*.txt")
    else:
        label_files = [
            (label_dir / f"{str(sample_id).zfill(6)}.txt") for sample_id in sample_ids
        ]
    return [
        get_objects_from_label(label_file, as_dict=True) for label_file in label_files
    ]


def evaluate(label_path, result_path, label_split_file, classes=(0,)):
    dt_annos = get_label_annos(result_path)
    if label_split_file is None:
        gt_sample_ids = None
    else:
        with open(label_split_file, "r") as f:
            lines = f.readlines()
        gt_sample_ids = [int(line) for line in lines]
    gt_annos = get_label_annos(label_path, gt_sample_ids)
    res_str = get_official_eval_result(gt_annos, dt_annos, classes)
    return res_str


def main():
    parser = ArgumentParser(description="arg parser")
    parser.add_argument("label_path", type=str)
    parser.add_argument("result_path", type=str)
    parser.add_argument("--label_split", type=str)
    parser.add_argument("--classes", type=int, default=0)
    args = parser.parse_args()
    print(args.label_path, args.result_path)
    print(evaluate(args.label_path, args.result_path, args.label_split, args.classes))


if __name__ == "__main__":
    main()
