from argparse import ArgumentParser
from configparser import ConfigParser
from pathlib import Path

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from datasets.kitti_dataset import (KittiDataset, collate_fn,
                                    write_kitti_detection_results)
from datasets.voxelization import VoxelGeneratorWrapper
from detection.voxel_rcnn.utils import rotate_points_bev
from detection.voxel_rcnn.voxel_rcnn import VoxelRCNN


def model_inference(
    root_path,
    output_dir,
    model,
    weight_path,
    voxel_size,
    point_cloud_range,
    num_point_features,
    max_num_points_per_voxel,
    max_num_voxels,
    class_names,
    batch_size,
    test_time_aug_cfgs=None,
):
    weights = torch.load(weight_path)

    model.load_state_dict(weights)
    model.eval()
    model.cuda()

    voxel_generator = VoxelGeneratorWrapper(
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        num_point_features=num_point_features,
        max_num_points_per_voxel=max_num_points_per_voxel,
        max_num_voxels=max_num_voxels,
    )

    for seq in range(21):
        seq = str(seq).zfill(4)

        if test_time_aug_cfgs is None:
            dataset = KittiDataset(
                root_path=root_path,
                seq=seq,
                voxel_generator=voxel_generator,
                point_cloud_range=point_cloud_range,
                aug_cfg=None,
            )
            data_loader = DataLoader(
                dataset, batch_size=batch_size, collate_fn=collate_fn
            )
            pbar = tqdm.tqdm(data_loader)
            for batch_dict in pbar:
                for key, value in batch_dict.items():
                    if key not in ["frame_id", "batch_size"]:
                        batch_dict[key] = torch.from_numpy(value).float().cuda()
                with torch.no_grad():
                    batch_dict = model.forward(batch_dict)
                pred_dicts = model.post_processing(batch_dict)
                write_kitti_detection_results(
                    pred_dicts,
                    class_names=class_names,
                    frame_ids=batch_dict["frame_id"],
                    calibrations=dataset.calib,
                    img_hw_dict=dataset.img_hw_dict,
                    output_dir=output_dir / seq,
                )
        else:
            # no tta
            seq_middle_dir = output_dir / "middle_data" / "no_aug" / seq
            seq_middle_dir.mkdir(exist_ok=True, parents=True)
            dataset = KittiDataset(
                root_path=root_path,
                seq=seq,
                voxel_generator=voxel_generator,
                point_cloud_range=point_cloud_range,
                aug_cfg=None,
            )
            data_loader = DataLoader(dataset, collate_fn=collate_fn)
            pbar = tqdm.tqdm(data_loader)
            for batch_dict in pbar:
                pbar.set_description(f"{seq}-no_aug")
                for key, value in batch_dict.items():
                    if key not in ["frame_id", "batch_size"]:
                        batch_dict[key] = torch.from_numpy(value).float().cuda()
                with torch.no_grad():
                    batch_dict = model.forward(batch_dict)
                pred_dict = {
                    "batch_size": batch_dict["batch_size"],
                    "batch_box_preds": batch_dict["batch_box_preds"],
                    "batch_cls_preds": batch_dict["batch_cls_preds"],
                    "frame_ids": batch_dict["frame_id"],
                    "calib": dataset.calib,
                    "class_names": class_names,
                    "aug_cfg": None,
                }
                if len(class_names) > 1:
                    pred_dict["roi_labels"] = batch_dict["roi_labels"]
                torch.save(
                    pred_dict, seq_middle_dir / f"{'_'.join(batch_dict['frame_id'])}.pt"
                )

            # tta
            for aug_name, aug_params in test_time_aug_cfgs.items():
                for aug_idx, aug_param in enumerate(aug_params):
                    aug_cfg = (aug_name, aug_param)
                    seq_middle_dir = (
                        output_dir / "middle_data" / aug_name / str(aug_idx) / seq
                    )
                    seq_middle_dir.mkdir(exist_ok=True, parents=True)

                    dataset = KittiDataset(
                        root_path=root_path,
                        seq=seq,
                        voxel_generator=voxel_generator,
                        point_cloud_range=point_cloud_range,
                        aug_cfg=aug_cfg,
                    )
                    data_loader = DataLoader(dataset, collate_fn=collate_fn)
                    pbar = tqdm.tqdm(data_loader)
                    for batch_dict in pbar:
                        pbar.set_description(f"{seq}-{aug_name}-{aug_idx}")
                        for key, value in batch_dict.items():
                            if key not in ["frame_id", "batch_size"]:
                                batch_dict[key] = torch.from_numpy(value).float().cuda()
                        with torch.no_grad():
                            batch_dict = model.forward(batch_dict)
                        pred_dict = {
                            "batch_size": batch_dict["batch_size"],
                            "batch_box_preds": batch_dict["batch_box_preds"],
                            "batch_cls_preds": batch_dict["batch_cls_preds"],
                            "frame_ids": batch_dict["frame_id"],
                            "calib": dataset.calib,
                            "class_names": class_names,
                            "aug_cfg": aug_cfg,
                        }
                        if len(class_names) > 1:
                            pred_dict["roi_labels"] = batch_dict["roi_labels"]
                        torch.save(
                            pred_dict,
                            seq_middle_dir / f"{'_'.join(batch_dict['frame_id'])}.pt",
                        )


def reverse_aug_boxes(data_dict):
    boxes = data_dict["batch_box_preds"]
    aug_method, aug_param = data_dict["aug_cfg"]
    if aug_method == "scale":
        boxes[:, :, :6] /= aug_param
    elif aug_method == "rotate":
        boxes = rotate_points_bev(
            boxes, boxes.new_tensor(-aug_param).repeat(len(boxes), 1)
        )
        boxes[:, :, 6] -= aug_param
    elif aug_method == "translate":
        boxes[:, :, :3] -= boxes.new_tensor(aug_param)
    elif aug_method == "flip":
        boxes[:, :, 1] *= -1
        boxes[:, :, 6] *= -1
    data_dict["batch_box_preds"] = boxes
    return data_dict


def tta_fusion(
    root_path, output_dir: Path, model, class_names, test_time_aug_cfgs: dict
):
    for seq in range(21):
        seq = str(seq).zfill(4)
        middle_dir = output_dir / "middle_data"
        no_aug_dir = middle_dir / "no_aug" / seq
        dataset = KittiDataset(
            root_path=root_path,
            seq=seq,
            voxel_generator=None,
            point_cloud_range=None,
            aug_cfg=None,
        )
        pbar = tqdm.tqdm(list(no_aug_dir.iterdir()))
        pbar.set_description(f"fusing {seq} data")
        for filepath in pbar:
            filename = filepath.name
            data_dicts = [torch.load(filepath)]

            for aug_name, aug_params in test_time_aug_cfgs.items():
                for aug_idx in range(len(aug_params)):
                    cur_aug_data = torch.load(
                        middle_dir / aug_name / str(aug_idx) / seq / filename
                    )
                    cur_aug_data = reverse_aug_boxes(cur_aug_data)
                    data_dicts.append(cur_aug_data)

            batch_box_preds = torch.cat(
                [d["batch_box_preds"] for d in data_dicts], dim=1
            )
            batch_cls_preds = torch.cat(
                [d["batch_cls_preds"] for d in data_dicts], dim=1
            )
            data_dict = data_dicts[0]
            data_dict["batch_box_preds"] = batch_box_preds
            data_dict["batch_cls_preds"] = batch_cls_preds
            pred_dicts = model.post_processing(data_dict)

            write_kitti_detection_results(
                pred_dicts,
                class_names=class_names,
                frame_ids=data_dicts[0]["frame_ids"],
                calibrations=dataset.calib,
                img_hw_dict=dataset.img_hw_dict,
                output_dir=output_dir / seq,
            )


def main():
    parser = ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("split", type=str)
    parser.add_argument("--inference", action="store_true")
    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config)

    root_dir = Path(config["data"]["root_dir"]) / args.split
    det3d_name = config["detection"]["det3d_name"]
    output_dir = root_dir / "det3d_out" / det3d_name
    weight_path = config["detection"]["det3d_ckpt"]
    batch_size = config["detection"].getint("batch_size")
    score_thresh = config["detection"].getfloat("score_thresh")

    tta = config["detection"].getboolean("tta")
    test_time_aug_cfgs = (
        {
            "rotate": [np.pi / 4, -np.pi / 4, np.pi / 8, -np.pi / 8],
            "scale": [0.95, 1.05],
            "translate": [[-5, 0, 0], [5, 0, 0], [0, -5, 0], [0, 5, 0]],
            "flip": [None],
        }
        if tta
        else None
    )

    point_cloud_range = [0, -40, -3, 70.4, 40, 1]
    voxel_size = [0.05, 0.05, 0.1]
    num_point_features = 4
    max_num_points_per_voxel = 5
    max_num_voxels = 40000
    class_names = ["Car"]

    model = VoxelRCNN(
        input_channels=num_point_features,
        num_classes=len(class_names),
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        score_thresh=score_thresh,
    )

    if args.inference:
        model_inference(
            root_dir,
            output_dir,
            model,
            weight_path,
            voxel_size,
            point_cloud_range,
            num_point_features,
            max_num_points_per_voxel,
            max_num_voxels,
            class_names,
            batch_size,
            test_time_aug_cfgs,
        )

    if tta:
        tta_fusion(root_dir, output_dir, model, class_names, test_time_aug_cfgs)


if __name__ == "__main__":
    main()
