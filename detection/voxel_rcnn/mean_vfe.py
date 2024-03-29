import torch


class MeanVFE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch_dict):
        voxel_features, voxel_num_points = (
            batch_dict["voxels"],
            batch_dict["voxel_num_points"],
        )
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(
            voxel_features
        )
        points_mean = points_mean / normalizer  # (num_voxels, C)
        batch_dict["voxel_features"] = points_mean.contiguous()
        return batch_dict
