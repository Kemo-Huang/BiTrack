# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from pathlib import Path

import mmcv
import numpy as np
import torch
import tqdm
from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS
from PIL import Image


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("root_dir", help="Root dir")
    parser.add_argument("name", help="model name")
    parser.add_argument("config", help="Config file")
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    root_dir = Path(args.root_dir)
    img_dir = root_dir / "image_02"
    assert img_dir.exists()
    det_out_dir = root_dir / "det2d_out" / args.name
    seg_out_dir = root_dir / "seg_out" / args.name

    if args.visualize:
        visualizer = VISUALIZERS.build(model.cfg.visualizer)
        visualizer.dataset_meta = model.dataset_meta

    for seq in range(21):
        seq = str(seq).zfill(4)
        pbar = tqdm.tqdm(list((img_dir / seq).iterdir()))
        pbar.set_description(seq)
        for img_file in pbar:
            image = mmcv.imread(img_file, channel_order="rgb")
            result = inference_detector(model, image)
            if args.visualize:
                visualizer.add_datasample(
                    "result",
                    image,
                    data_sample=result,
                    draw_gt=None,
                    wait_time=0,
                )
                visualizer.show()
            boxes = result.pred_instances.bboxes.cpu()
            labels = result.pred_instances.labels.cpu()
            scores = result.pred_instances.scores.cpu()
            masks = result.pred_instances.masks.cpu()

            class_mask = torch.isin(
                labels, torch.tensor([2, 5, 7])
            )  # cars, buses, trucks
            boxes = boxes[class_mask].tolist()
            scores = scores[class_mask].tolist()
            labels = labels[class_mask].tolist()
            masks = masks[class_mask].numpy()

            lines = []
            seg_img = np.zeros(image.shape[:2], dtype=np.uint8)
            for i in range(len(boxes)):
                lines.append(
                    " ".join([str(x) for x in boxes[i]]) + f" {scores[i]} {labels[i]}\n"
                )
                seg_img[masks[i]] = i + 1

            seq_det_out_dir = det_out_dir / seq
            seq_det_out_dir.mkdir(exist_ok=True, parents=True)

            seq_seg_out_dir = seg_out_dir / seq
            seq_seg_out_dir.mkdir(exist_ok=True, parents=True)

            with open(seq_det_out_dir / f"{img_file.stem}.txt", "w") as f:
                f.writelines(lines)

            Image.fromarray(seg_img).save(seq_seg_out_dir / f"{img_file.stem}.png")


if __name__ == "__main__":
    args = parse_args()
    main(args)
