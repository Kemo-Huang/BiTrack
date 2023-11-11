import numpy as np
import torch
from torch.nn import Module

from .anchor_head_single import AnchorHeadSingle
from .base_bev_backbone import BaseBEVBackbone
from .height_compression import HeightCompression
from .mean_vfe import MeanVFE
from .spconv_backbone import VoxelBackBone8x
from .utils import class_agnostic_nms
from .voxel_rcnn_head import VoxelRCNNHead


class VoxelRCNN(Module):
    def __init__(
        self,
        input_channels,
        num_classes,
        point_cloud_range,
        voxel_size,
        score_thresh=0.2,
    ):
        super().__init__()
        if not isinstance(point_cloud_range, np.ndarray):
            point_cloud_range = np.array(point_cloud_range)
        if not isinstance(voxel_size, np.ndarray):
            voxel_size = np.array(voxel_size)
        grid_size = (point_cloud_range[3:6] - point_cloud_range[0:3]) / voxel_size
        self.grid_size = np.round(grid_size).astype(np.int64)
        self.num_classes = num_classes
        self.score_thresh = score_thresh

        self.backbone_3d = VoxelBackBone8x(
            input_channels=input_channels, grid_size=grid_size
        )
        self.backbone_2d = BaseBEVBackbone(
            model_cfg={
                "LAYER_NUMS": [5, 5],
                "LAYER_STRIDES": [1, 2],
                "NUM_FILTERS": [64, 128],
                "UPSAMPLE_STRIDES": [1, 2],
                "NUM_UPSAMPLE_FILTERS": [128, 128],
            },
            input_channels=256,
        )
        self.dense_head = AnchorHeadSingle(
            model_cfg={
                "USE_DIRECTION_CLASSIFIER": True,
                "NUM_DIR_BINS": 2,
                "DIR_OFFSET": 0.78539,
                "DIR_LIMIT_OFFSET": 0.0,
                "ANCHOR_GENERATOR_CONFIG": [
                    {
                        "class_name": "Car",
                        "anchor_sizes": [[3.9, 1.6, 1.56]],
                        "anchor_rotations": [0, 1.57],
                        "anchor_bottom_heights": [-1.78],
                        "align_center": False,
                        "feature_map_stride": 8,
                        "matched_threshold": 0.6,
                        "unmatched_threshold": 0.45,
                    },
                    {
                        "class_name": "Pedestrian",
                        "anchor_sizes": [[0.8, 0.6, 1.73]],
                        "anchor_rotations": [0, 1.57],
                        "anchor_bottom_heights": [-0.6],
                        "align_center": False,
                        "feature_map_stride": 8,
                        "matched_threshold": 0.5,
                        "unmatched_threshold": 0.35,
                    },
                    {
                        "class_name": "Cyclist",
                        "anchor_sizes": [[1.76, 0.6, 1.73]],
                        "anchor_rotations": [0, 1.57],
                        "anchor_bottom_heights": [-0.6],
                        "align_center": False,
                        "feature_map_stride": 8,
                        "matched_threshold": 0.5,
                        "unmatched_threshold": 0.35,
                    },
                ],
            },
            input_channels=self.backbone_2d.num_bev_features,
            num_classes=num_classes,
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
        )
        self.roi_head = VoxelRCNNHead(
            backbone_channels=self.backbone_3d.backbone_channels,
            model_cfg={
                "SHARED_FC": [256, 256],
                "CLS_FC": [256, 256],
                "REG_FC": [256, 256],
                "DP_RATIO": 0.3,
                "NMS_CONFIG": {
                    "NMS_TYPE": "nms_gpu",
                    "USE_FAST_NMS": False,
                    "SCORE_THRESH": 0.0,
                    "NMS_PRE_MAXSIZE": 2048,
                    "NMS_POST_MAXSIZE": 100,
                    "NMS_THRESH": 0.7,
                },
                "ROI_GRID_POOL": {
                    "FEATURES_SOURCE": ["x_conv2", "x_conv3", "x_conv4"],
                    "PRE_MLP": True,
                    "GRID_SIZE": 6,
                    "POOL_LAYERS": {
                        "x_conv2": {
                            "MLPS": [[32, 32]],
                            "QUERY_RANGES": [[4, 4, 4]],
                            "POOL_RADIUS": [0.4],
                            "NSAMPLE": [16],
                            "POOL_METHOD": "max_pool",
                        },
                        "x_conv3": {
                            "MLPS": [[32, 32]],
                            "QUERY_RANGES": [[4, 4, 4]],
                            "POOL_RADIUS": [0.8],
                            "NSAMPLE": [16],
                            "POOL_METHOD": "max_pool",
                        },
                        "x_conv4": {
                            "MLPS": [[32, 32]],
                            "QUERY_RANGES": [[4, 4, 4]],
                            "POOL_RADIUS": [1.6],
                            "NSAMPLE": [16],
                            "POOL_METHOD": "max_pool",
                        },
                    },
                },
            },
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
        )

        self.module_list = [
            MeanVFE(),
            self.backbone_3d,
            HeightCompression(),
            self.backbone_2d,
            self.dense_head,
            self.roi_head,
        ]

    def forward(self, batch_dict: dict):
        """
        batch_dict:
            voxels: (num_voxels, max_points_per_voxel, C)
            voxel_num_points: optional (num_voxels)
        """
        for module in self.module_list:
            batch_dict = module(batch_dict)
        return batch_dict

    def post_processing(self, batch_dict):
        batch_size = batch_dict["batch_size"]
        pred_dicts = []
        for batch_idx in range(batch_size):
            box_preds = batch_dict["batch_box_preds"][batch_idx]
            cls_preds = batch_dict["batch_cls_preds"][batch_idx]

            cls_preds = torch.sigmoid(cls_preds)
            cls_preds, label_preds = torch.max(cls_preds, dim=-1)

            if self.num_classes > 1:
                label_preds = batch_dict["roi_labels"][batch_idx]
            else:
                label_preds = label_preds + 1

            selected, selected_scores = class_agnostic_nms(
                box_scores=cls_preds,
                box_preds=box_preds,
                nms_config={
                    "NMS_TYPE": "nms_gpu",
                    "NMS_THRESH": 0.1,
                    "NMS_PRE_MAXSIZE": 4096,
                    "NMS_POST_MAXSIZE": 500,
                },
                score_thresh=self.score_thresh,
            )

            final_scores = selected_scores
            final_labels = label_preds[selected]
            final_boxes = box_preds[selected]

            record_dict = {
                "pred_boxes": final_boxes,
                "pred_scores": final_scores,
                "pred_labels": final_labels,
            }
            pred_dicts.append(record_dict)

        return pred_dicts
