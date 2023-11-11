import torch
from torch import nn

from .anchor_generator import AnchorGenerator
from .box_coder import ResidualCoder


def limit_period(val, offset=0.5, period=torch.pi):
    return val - torch.floor(val / period + offset) * period


class AnchorHeadSingle(nn.Module):
    def __init__(
        self,
        model_cfg,
        input_channels,
        num_classes,
        grid_size,
        point_cloud_range,
        **kwargs
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_classes = num_classes
        self.use_multihead = self.model_cfg.get("USE_MULTIHEAD", False)

        self.box_coder = ResidualCoder()

        anchors, self.num_anchors_per_location = self.generate_anchors(
            self.model_cfg["ANCHOR_GENERATOR_CONFIG"][:num_classes],
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            anchor_ndim=self.box_coder.code_size,
        )
        self.anchors = [x.cuda() for x in anchors]

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels,
            self.num_anchors_per_location * self.num_classes,
            kernel_size=1,
        )
        self.conv_box = nn.Conv2d(
            input_channels,
            self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1,
        )

        if "USE_DIRECTION_CLASSIFIER" in self.model_cfg:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg["NUM_DIR_BINS"],
                kernel_size=1,
            )
        else:
            self.conv_dir_cls = None

    @staticmethod
    def generate_anchors(
        anchor_generator_cfg, grid_size, point_cloud_range, anchor_ndim=7
    ):
        anchor_generator = AnchorGenerator(
            anchor_range=point_cloud_range, anchor_generator_config=anchor_generator_cfg
        )
        feature_map_size = [
            grid_size[:2] // config["feature_map_stride"]
            for config in anchor_generator_cfg
        ]
        anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(
            feature_map_size
        )

        if anchor_ndim != 7:
            for idx, anchors in enumerate(anchors_list):
                pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
                anchors_list[idx] = new_anchors

        return anchors_list, num_anchors_per_location_list

    def generate_predicted_boxes(
        self, batch_size, cls_preds, box_preds, dir_cls_preds=None
    ):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [
                        anchor.permute(3, 4, 0, 1, 2, 5)
                        .contiguous()
                        .view(-1, anchor.shape[-1])
                        for anchor in self.anchors
                    ],
                    dim=0,
                )
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        batch_cls_preds = (
            cls_preds.view(batch_size, num_anchors, -1).float()
            if not isinstance(cls_preds, list)
            else cls_preds
        )
        batch_box_preds = (
            box_preds.view(batch_size, num_anchors, -1)
            if not isinstance(box_preds, list)
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        )
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)

        if dir_cls_preds is not None:
            dir_offset = self.model_cfg["DIR_OFFSET"]
            dir_limit_offset = self.model_cfg["DIR_LIMIT_OFFSET"]
            dir_cls_preds = (
                dir_cls_preds.view(batch_size, num_anchors, -1)
                if not isinstance(dir_cls_preds, list)
                else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
            )
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

            period = 2 * torch.pi / self.model_cfg["NUM_DIR_BINS"]
            dir_rot = limit_period(
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            )
            batch_box_preds[..., 6] = (
                dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)
            )

        return batch_cls_preds, batch_box_preds

    def forward(self, batch_dict):
        spatial_features_2d = batch_dict["spatial_features_2d"]

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
        else:
            dir_cls_preds = None

        batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
            batch_size=batch_dict["batch_size"],
            cls_preds=cls_preds,
            box_preds=box_preds,
            dir_cls_preds=dir_cls_preds,
        )
        batch_dict["batch_cls_preds"] = batch_cls_preds
        batch_dict["batch_box_preds"] = batch_box_preds

        return batch_dict
