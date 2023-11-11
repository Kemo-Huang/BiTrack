from torch.nn import Module


class HeightCompression(Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch_dict):
        encoded_spconv_tensor = batch_dict["encoded_spconv_tensor"]
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict["spatial_features"] = spatial_features
        batch_dict["spatial_features_stride"] = batch_dict[
            "encoded_spconv_tensor_stride"
        ]
        return batch_dict
