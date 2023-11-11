import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor


def img_to_float_tensor(img, cuda=False):
    if isinstance(img, Image.Image):
        img = pil_to_tensor(img)
    elif isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    elif isinstance(img, torch.Tensor):
        pass
    else:
        img = pil_to_tensor(Image.open(img))
    img = img.float()
    if cuda:
        img = img.cuda()
    return img

class PointFeatFuse3P(nn.Module):
    # three path
    def __init__(self, num_points=250, ic=7, oc=64, maxpool=True):
        super(PointFeatFuse3P, self).__init__()
        self.conv1 = torch.nn.Conv1d(2, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(ic, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)
        self.e_conv3 = torch.nn.Conv1d(128, 256, 1)

        self.t_conv1 = torch.nn.Conv1d(3, 64, 1)
        self.t_conv2 = torch.nn.Conv1d(64, 128, 1)
        self.t_conv3 = torch.nn.Conv1d(128, 128, 1)

        self.conv4 = torch.nn.Conv1d(512, 256, 1)
        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, oc, 1)

        if maxpool:
            self.pool = torch.nn.MaxPool1d(num_points)
        else:
            self.pool = torch.nn.AvgPool1d(num_points)

        self.num_points = num_points

    def forward(self, x, emb, t, withInd=False):
        x = F.leaky_relu(self.conv1(x))
        emb = F.leaky_relu(self.e_conv1(emb))
        t = F.leaky_relu(self.t_conv1(t))

        x = F.leaky_relu(self.conv2(x))
        emb = F.leaky_relu(self.e_conv2(emb))
        t = F.leaky_relu(self.t_conv2(t))

        x = F.leaky_relu(self.conv3(x))
        emb = F.leaky_relu(self.e_conv3(emb))
        t = F.leaky_relu(self.t_conv3(t))

        pointfeat_2 = torch.cat((x, emb, t), dim=1)

        x1 = F.leaky_relu(self.conv4(pointfeat_2))
        x1 = F.leaky_relu(self.conv5(x1))
        x1 = F.leaky_relu(self.conv6(x1))
        if withInd:
            return self.pool(x1).squeeze(-1), torch.max(x1, dim=2)[1]
        return self.pool(x1).squeeze(-1)


class PoseNetFeatOffsetEmb(nn.Module):
    # bn with border
    def __init__(self, num_points=250, ic=7, border_points=200, border_ic=6, output_dim=64, category=False):
        super(PoseNetFeatOffsetEmb, self).__init__()
        self.category = category
        bc = 256
        self.borderConv = PointFeatFuse3P(ic=border_ic, oc=bc, num_points=border_points)

        self.conv1 = torch.nn.Conv1d(2, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.conv1_bn = nn.BatchNorm1d(64)
        self.conv2_bn = nn.BatchNorm1d(128)
        self.conv3_bn = nn.BatchNorm1d(256)

        self.e_conv1 = torch.nn.Conv1d(ic, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)
        self.e_conv3 = torch.nn.Conv1d(128, 256, 1)
        self.e_conv1_bn = nn.BatchNorm1d(64)
        self.e_conv2_bn = nn.BatchNorm1d(128)
        self.e_conv3_bn = nn.BatchNorm1d(256)

        self.conv4 = torch.nn.Conv1d(512, 256, 1)
        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 64, 1)
        self.conv4_bn = nn.BatchNorm1d(256)
        self.conv5_bn = nn.BatchNorm1d(512)

        self.conv7 = torch.nn.Conv1d(512, 256, 1)
        self.conv8 = torch.nn.Conv1d(256, 512, 1)
        self.conv9 = torch.nn.Conv1d(512, 64, 1)
        self.conv7_bn = nn.BatchNorm1d(256)
        self.conv8_bn = nn.BatchNorm1d(512)

        self.conv_weight = torch.nn.Conv1d(128, 1, 1)

        self.last_emb = nn.Sequential(
            nn.Linear(704+bc, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_dim)
        )
        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.mp2 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points

    def forward(self, inp, borders, spatialEmbs, with_weight=False):
        x, emb = inp[:,-2:], inp[:,:-2]
        x = F.leaky_relu(self.conv1_bn(self.conv1(x)))
        emb = F.leaky_relu(self.e_conv1_bn(self.e_conv1(emb)))

        x = F.leaky_relu(self.conv2_bn(self.conv2(x)))
        emb = F.leaky_relu(self.e_conv2_bn(self.e_conv2(emb)))

        x = F.leaky_relu(self.conv3_bn(self.conv3(x)))          # B,256,N
        emb = F.leaky_relu(self.e_conv3_bn(self.e_conv3(emb)))  # B,256,N

        pointfeat_2 = torch.cat((x, emb), dim=1)

        x1 = F.leaky_relu(self.conv4_bn(self.conv4(pointfeat_2)))
        x1 = F.leaky_relu(self.conv5_bn(self.conv5(x1)))
        x1 = F.leaky_relu(self.conv6(x1))                       # B,64,N
        ap_x1 = self.ap1(x1).squeeze(-1)                        # B,64

        x2 = F.leaky_relu(self.conv7_bn(self.conv7(pointfeat_2)))
        x2 = F.leaky_relu(self.conv8_bn(self.conv8(x2)))
        x2 = F.leaky_relu(self.conv9(x2))                       # B,64,N
        mp_x2 = self.mp2(x2).squeeze(-1)                        # B,64

        weightFeat = self.conv_weight(torch.cat([x1, x2], dim=1))   #B,1,N
        weight = torch.nn.Softmax(2)(weightFeat)
        weight_x3 = (weight.expand_as(pointfeat_2) * pointfeat_2).sum(2)

        if with_weight:
            border_feat, bg_inds = self.borderConv(borders[:, 3:5], borders[:, :3], borders[:, 5:], withInd=with_weight)
            x = torch.cat([ap_x1, mp_x2, weight_x3, border_feat, spatialEmbs], dim=1)
            outp = self.last_emb(x)
            return outp, weight, bg_inds
        else:
            border_feat = self.borderConv(borders[:, 3:5], borders[:, :3], borders[:, 5:])

        x = torch.cat([ap_x1, mp_x2, weight_x3, border_feat, spatialEmbs], dim=1)
        outp = self.last_emb(x)
        return outp


class EmbedNet(nn.Module):
    # for uv offset and category
    def __init__(self, num_points, border_ic, env_points, outputD, margin=0.2):
        super().__init__()
        self.num_points = num_points
        self.env_points = env_points
        self.point_feat = PoseNetFeatOffsetEmb(num_points=num_points, ic=3, border_points=env_points, border_ic=border_ic, output_dim=outputD, category=True)
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.category_embedding = torch.FloatTensor([
            [0.9479751586914062, 0.4561353325843811, 0.16707628965377808],
            [0.5455077290534973, -0.6193588972091675, -2.629554510116577],
        ])


    def location_embedding(self, xyxys, dim_g=64, wave_len=1000):
        x_min, y_min, x_max, y_max = torch.chunk(xyxys, 4, dim=1)
        cx = (x_min + x_max) * 0.5
        cy = (y_min + y_max) * 0.5
        w = (x_max - x_min) + 1.
        h = (y_max - y_min) + 1.
        position_mat = torch.cat((cx, cy, w, h), -1)

        feat_range = torch.arange(dim_g / 8).cuda()
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        dim_mat = dim_mat.view(1, 1, -1)
        position_mat = position_mat.view(xyxys.shape[0], 4, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(xyxys.shape[0], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)
        return embedding

    def compute_triplet_loss(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        loss = torch.zeros([1]).cuda()
        if mask.float().unique().shape[0] > 1:
            dist_ap, dist_an = [], []
            for i in range(n):
                dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
                dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
            dist_ap = torch.cat(dist_ap)
            dist_an = torch.cat(dist_an)
            # Compute ranking hinge loss
            y = torch.ones_like(dist_an)
            loss = self.ranking_loss(dist_an, dist_ap, y).unsqueeze(0)
        return loss

    def forward(self, fg_points, bg_points, xyxys, labels=None, infer=True, visualize=False):
        embeds = self.location_embedding(xyxys)
        if infer:
            embeds = self.point_feat(fg_points, bg_points, embeds)
            return embeds
        elif visualize:
            embeds, point_weights, bg_inds = self.point_feat(fg_points, bg_points, embeds, with_weight=True)
            return embeds, point_weights, bg_inds
        else:
            embeds = self.point_feat(fg_points, bg_points, embeds)
            labels = labels[0]
            return self.compute_triplet_loss(embeds, labels)
    
    def preprocessing(self, img, instance_map, instance_ids=None, cuda=False):
        img = img_to_float_tensor(img, cuda) / 255
        instance_map = img_to_float_tensor(instance_map, cuda)

        if len(instance_map.shape) == 3:
            instance_map = instance_map[0]
        
        fg_points = []
        bg_points = []
        xyxys = []
        masks = []
        
        h, w = img.shape[-2:]
        label = (instance_map > 0).long()
        if instance_ids is None:
            instance_ids = torch.unique(instance_map, sorted=True)[1:]
        
        for inst_id in instance_ids:
            assert inst_id is not None
            mask = (instance_map == inst_id)
            masks.append(mask[None])
            mask_inds = torch.nonzero(mask)
            min_val, _ = mask_inds.min(dim=0)
            y0, x0 = min_val
            max_val, _ = mask_inds.max(dim=0)
            y1, x1 = max_val
            xyxys.append(torch.stack([x0/w, y0/h, x1/w, y1/h])[None])

            fg_choice = torch.randint(low=0, high=len(mask_inds), size=(
                self.num_points,), dtype=mask_inds.dtype, device=mask_inds.device)
            mask_inds = mask_inds[fg_choice]

            fg_rgb_feat = img[:, mask_inds[:, 0], mask_inds[:, 1]] / 255  # (3, N)

            center_ind = mask_inds.float().mean(dim=0)
            fg_ind_feat = (mask_inds - center_ind) / 128
            fg_ind_feat = torch.transpose(fg_ind_feat, 0, 1).contiguous()  # (2, N)

            fg_points.append(torch.cat((fg_rgb_feat, fg_ind_feat), dim=0)[None])

            # Crops image patch including background
            # Enlarge patch by 0.4
            y_pad = (0.2 * (y1 - y0)).int()
            x_pad = (0.2 * (x1 - x0)).int()
            bg_mask_ind = torch.nonzero(~mask[
                torch.clamp_min(y0 - y_pad, 0):torch.clamp_max(y1 + y_pad + 1, h),
                torch.clamp_min(x0 - x_pad, 0):torch.clamp_max(x1 + x_pad + 1, w)
            ])
            if len(bg_mask_ind) == 0:
                bg_points.append(img.new_zeros((1, 8, self.env_points)))
            else:
                bg_choice = torch.randint(low=0, high=len(bg_mask_ind), size=(
                    self.env_points,), dtype=mask_inds.dtype, device=mask_inds.device)
                bg_mask_ind = bg_mask_ind[bg_choice]

                bg_rgb_feat = img[:, bg_mask_ind[:, 0],
                                bg_mask_ind[:, 1]] / 255  # (3, N)

                bg_ind_feat = (bg_mask_ind - center_ind) / 128
                bg_ind_feat = torch.transpose(
                    bg_ind_feat, 0, 1).contiguous()  # (2, N)

                cat_embds = self.category_embedding[label[
                    bg_mask_ind[:, 0], bg_mask_ind[:, 1]
                ]]
                cat_embds = torch.transpose(cat_embds, 0, 1).contiguous()  # (3, N)

                bg_points.append(
                    torch.cat((bg_rgb_feat, bg_ind_feat, cat_embds), dim=0)[None])
        
        if len(xyxys) > 0:
            fg_points = torch.cat(fg_points)  # (num_instance, 5, num_points)
            bg_points = torch.cat(bg_points)  # (num_instance, 8, num_points)
            xyxys = torch.cat(xyxys)  # (num_instance, 4)
            masks = torch.cat(masks)  # (num_instance)
        
        return fg_points, bg_points, xyxys, masks

