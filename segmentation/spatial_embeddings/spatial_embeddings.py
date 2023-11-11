# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.functional as F


class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(
            ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True
        )
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(
            chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True
        )

        self.conv1x3_1 = nn.Conv2d(
            chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True
        )

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(
            chann,
            chann,
            (3, 1),
            stride=1,
            padding=(1 * dilated, 0),
            bias=True,
            dilation=(dilated, 1),
        )

        self.conv1x3_2 = nn.Conv2d(
            chann,
            chann,
            (1, 3),
            stride=1,
            padding=(0, 1 * dilated),
            bias=True,
            dilation=(1, dilated),
        )

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, x):
        output = self.conv3x1_1(x)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if self.dropout.p != 0:
            output = self.dropout(output)

        return F.relu(output + x)  # +input = identity (residual connection)


class Encoder(nn.Module):
    def __init__(self, num_classes, input_channel=3):
        super().__init__()
        self.initial_block = DownsamplerBlock(input_channel, 16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))

        for _ in range(5):  # 5 times
            self.layers.append(non_bottleneck_1d(64, 0.03, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        for _ in range(2):  # 2 times
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))

        # Only in encoder mode:
        self.output_conv = nn.Conv2d(
            128, num_classes, 1, stride=1, padding=0, bias=True
        )

    def forward(self, x, predict=False, feat=False):
        output = self.initial_block(x)
        feats = output

        for layer in self.layers:
            output = layer(output)

        if predict:
            att = self.output_conv(output)
            return output, att

        if feat:
            return output, feats
        return output


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True
        )
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(
            16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_conv(x)


class ERFNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = Encoder(num_classes)
        self.decoder = Decoder(num_classes)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder.forward(input, predict=False)
            return self.decoder.forward(output)


class BranchedERFNet(nn.Module):
    def __init__(self, num_classes, input_channel=3):
        super().__init__()
        self.encoder = Encoder(sum(num_classes), input_channel=input_channel)
        self.decoders = nn.ModuleList([Decoder(n) for n in num_classes])

        xm = torch.linspace(0, 2, 2048).view(1, 1, -1).expand(1, 1024, 2048)
        ym = torch.linspace(0, 1, 1024).view(1, -1, 1).expand(1, 1024, 2048)
        self.xym = torch.cat((xm, ym)).cuda()

    def init_output(self, n_sigma=1):
        with torch.no_grad():
            output_conv = self.decoders[0].output_conv
            print("initialize last layer with size: ", output_conv.weight.size())

            output_conv.weight[:, 0:2, :, :].fill_(0)
            output_conv.bias[0:2].fill_(0)

            output_conv.weight[:, 2 : 2 + n_sigma, :, :].fill_(0)
            output_conv.bias[2 : 2 + n_sigma].fill_(1)

    def forward(self, x, only_encode=False):
        if only_encode:
            return self.encoder.forward(x, predict=True)
        else:
            output = self.encoder.forward(x, predict=False)
            return torch.cat([decoder.forward(output) for decoder in self.decoders], 1)

    def cluster(
        self, prediction: torch.Tensor, n_sigma=1, threshold=0.5, min_pixel=128
    ) -> torch.Tensor:
        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        sigma = prediction[2 : 2 + n_sigma]  # n_sigma x h x w
        seed_map = torch.sigmoid(prediction[2 + n_sigma : 2 + n_sigma + 1])  # 1 x h x w

        instance_map = prediction.new_zeros((height, width), dtype=torch.uint8)

        count = 1
        mask = seed_map > 0.5
        num_pixel = mask.sum()

        if num_pixel > min_pixel:
            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(
                2, num_pixel
            )
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, num_pixel)
            seed_map_masked = seed_map[mask]

            unclustered = prediction.new_ones(num_pixel, dtype=torch.float)
            instance_map_masked = prediction.new_zeros(num_pixel, dtype=torch.uint8)

            while unclustered.sum() > min_pixel:
                seed_score, seed = torch.max(seed_map_masked * unclustered, dim=0)
                if seed_score < threshold:
                    break
                center = spatial_emb_masked[:, seed : seed + 1]
                unclustered[seed] = 0
                s = torch.exp(sigma_masked[:, seed : seed + 1] * 10)
                dist = torch.exp(
                    -1 * torch.sum(torch.pow(spatial_emb_masked - center, 2) * s, 0)
                )
                proposal = dist > 0.5
                num_proposals = proposal.sum()
                if num_proposals > min_pixel:
                    if unclustered[proposal].sum() / num_proposals > 0.5:
                        assert count < 256
                        instance_map_masked[proposal] = count
                        count += 1
                unclustered[proposal] = 0
            instance_map[mask.squeeze(0)] = instance_map_masked

        return instance_map
