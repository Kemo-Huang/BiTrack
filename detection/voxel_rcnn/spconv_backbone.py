from functools import partial

import spconv.pytorch as spconv
from torch import nn


def find_all_spconv_keys(model: nn.Module, prefix=""):
    found_keys = set()
    for name, child in model.named_children():
        new_prefix = f"{prefix}.{name}" if prefix != "" else name
        if isinstance(child, spconv.conv.SparseConvolution):
            new_prefix = f"{new_prefix}.weight"
            found_keys.add(new_prefix)
        found_keys.update(find_all_spconv_keys(child, prefix=new_prefix))
    return found_keys


def convert_model_weights_spconv(weights: dict, model):
    spconv_keys = find_all_spconv_keys(model)
    state_dict = model.state_dict()
    converted_weights = {}
    for key, val in weights.items():
        if (
            key in spconv_keys
            and key in state_dict
            and state_dict[key].shape != val.shape
        ):
            # with different spconv versions, we need to adapt weight shapes for spconv blocks
            # adapt spconv weights from version 1.x to version 2.x if you used weights from spconv 1.x

            val_native = val.transpose(
                -1, -2
            )  # (k1, k2, k3, c_in, c_out) to (k1, k2, k3, c_out, c_in)
            if val_native.shape == state_dict[key].shape:
                val = val_native.contiguous()
            else:
                assert val.shape.__len__() == 5, "currently only spconv 3D is supported"
                val_implicit = val.permute(
                    4, 0, 1, 2, 3
                )  # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
                if val_implicit.shape == state_dict[key].shape:
                    val = val_implicit.contiguous()

            converted_weights[key] = val
    weights.update(converted_weights)
    return weights


def post_act_block(
    in_channels,
    out_channels,
    kernel_size,
    indice_key=None,
    stride=1,
    padding=0,
    conv_type="subm",
    norm_fn=None,
):
    if conv_type == "subm":
        conv = spconv.SubMConv3d(
            in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key
        )
    elif conv_type == "spconv":
        conv = spconv.SparseConv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            indice_key=indice_key,
        )
    elif conv_type == "inverseconv":
        conv = spconv.SparseInverseConv3d(
            in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False
        )
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class VoxelBackBone8x(nn.Module):
    def __init__(self, input_channels, grid_size, **kwargs):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_channels, 16, 3, padding=1, bias=False, indice_key="subm1"
            ),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key="subm1"),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] -> [800, 704, 21]
            block(
                16,
                32,
                3,
                norm_fn=norm_fn,
                stride=2,
                padding=1,
                indice_key="spconv2",
                conv_type="spconv",
            ),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key="subm2"),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key="subm2"),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] -> [400, 352, 11]
            block(
                32,
                64,
                3,
                norm_fn=norm_fn,
                stride=2,
                padding=1,
                indice_key="spconv3",
                conv_type="spconv",
            ),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key="subm3"),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key="subm3"),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] -> [200, 176, 5]
            block(
                64,
                64,
                3,
                norm_fn=norm_fn,
                stride=2,
                padding=(0, 1, 1),
                indice_key="spconv4",
                conv_type="spconv",
            ),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key="subm4"),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key="subm4"),
        )

        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(
                64,
                128,
                (3, 1, 1),
                stride=(2, 1, 1),
                padding=0,
                bias=False,
                indice_key="spconv_down2",
            ),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            "x_conv1": 16,
            "x_conv2": 32,
            "x_conv3": 64,
            "x_conv4": 64,
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = (
            batch_dict["voxel_features"],
            batch_dict["voxel_coords"],
        )
        batch_size = batch_dict["batch_size"]
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size,
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update(
            {"encoded_spconv_tensor": out, "encoded_spconv_tensor_stride": 8}
        )
        batch_dict.update(
            {
                "multi_scale_3d_features": {
                    "x_conv1": x_conv1,
                    "x_conv2": x_conv2,
                    "x_conv3": x_conv3,
                    "x_conv4": x_conv4,
                }
            }
        )
        batch_dict.update(
            {
                "multi_scale_3d_strides": {
                    "x_conv1": 1,
                    "x_conv2": 2,
                    "x_conv3": 4,
                    "x_conv4": 8,
                }
            }
        )

        return batch_dict
