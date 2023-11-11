import torch

from .iou3d_nms import iou3d_nms_utils


def rotate_points_along_z(points, angles):
    """
    Args:
        points: (B, N, 3 + C)
        angles: (B), angles along z-axis, angle increases x ==> y
    Returns:

    """
    cosa = torch.cos(angles)
    sina = torch.sin(angles)
    zeros = angles.new_zeros(points.shape[0])
    ones = angles.new_ones(points.shape[0])
    rot_matrix = (
        torch.stack((cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones), dim=1)
        .view(-1, 3, 3)
        .float()
    )
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot


def rotate_points_bev(points, angles):
    # same as above
    cosa = torch.cos(angles)
    sina = torch.sin(angles)
    rot_matrix = (
        torch.stack(
            (
                cosa,
                sina,
                -sina,
                cosa,
            ),
            dim=1,
        )
        .view(-1, 2, 2)
        .float()
    )
    points_rot = torch.matmul(points[:, :, 0:2], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 2:]), dim=-1)
    return points_rot


def class_agnostic_nms(box_scores, box_preds, nms_config, score_thresh=None):
    src_box_scores = box_scores
    if score_thresh is not None:
        scores_mask = box_scores >= score_thresh
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]

    selected = []
    if box_scores.shape[0] > 0:
        box_scores_nms, indices = torch.topk(
            box_scores, k=min(nms_config["NMS_PRE_MAXSIZE"], box_scores.shape[0])
        )
        boxes_for_nms = box_preds[indices]
        keep_idx = getattr(iou3d_nms_utils, nms_config["NMS_TYPE"])(
            boxes_for_nms[:, 0:7],
            box_scores_nms,
            nms_config["NMS_THRESH"],
            **nms_config
        )
        selected = indices[keep_idx[: nms_config["NMS_POST_MAXSIZE"]]]

    if score_thresh is not None:
        original_idxs = scores_mask.nonzero().view(-1)
        selected = original_idxs[selected]
    return selected, src_box_scores[selected]


def boxes3d_to_bev(boxes: torch.Tensor):
    """
      0 ------ 1
      |        |
      |        |
      3 ------ 2

    Args:
        boxes (torch.Tensor): _description_

    Returns:
        _type_: _description_
    """
    template = boxes.new_tensor(([1, 1], [1, -1], [-1, -1], [-1, 1])) / 2

    corners_bev = rotate_points_bev(
        boxes[:, None, 3:5] * template[None, :, :], boxes[:, 6]
    )
    corners_bev += boxes[:, None, 0:2]
    return corners_bev


def weighted_box_fusion(
    box_scores, box_preds, box_labels, fusion_thresh=0.55, score_thresh=None, n_models=1
):
    if score_thresh is not None:
        scores_mask = box_scores >= score_thresh
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]
        box_labels = box_labels[scores_mask]

    num_boxes = box_scores.shape[0]
    if num_boxes == 0:
        return torch.zeros((0, 7)), torch.zeros(0), torch.zeros(0)
    else:
        box_preds = box_preds.view(num_boxes, box_preds.shape[-1])
        inds = torch.argsort(box_scores, descending=True)
        box_scores = box_scores[inds]
        box_preds = box_preds[inds]
        box_labels = box_labels[inds]

        cluster_boxes = [box_preds[0:1]]
        cluster_scores = [box_scores[0:1]]
        cluster_labels = [box_labels[0:1]]

        fused_boxes = [box_preds[0]]
        fused_scores = [box_scores[0]]
        fused_labels = [box_labels[0]]

        for i in range(1, num_boxes):
            bev_ious = iou3d_nms_utils.boxes_iou_bev(
                torch.stack(fused_boxes), box_preds[i : i + 1]
            ).view(-1)
            max_idx = torch.argmax(bev_ious)
            if bev_ious[max_idx] <= fusion_thresh:
                cluster_boxes.append(box_preds[i : i + 1])
                cluster_scores.append(box_scores[i : i + 1])
                cluster_labels.append(box_labels[i : i + 1])

                fused_boxes.append(box_preds[i])
                fused_scores.append(box_scores[i])
                fused_labels.append(box_labels[i])
            else:
                cluster_boxes[max_idx] = torch.cat(
                    (cluster_boxes[max_idx], box_preds[i : i + 1])
                )
                cluster_scores[max_idx] = torch.cat(
                    (cluster_scores[max_idx], box_scores[i : i + 1])
                )
                cluster_labels[max_idx] = torch.cat(
                    (cluster_labels[max_idx], box_labels[i : i + 1])
                )

                # fusion here
                fused_boxes[max_idx] = torch.sum(
                    cluster_scores[max_idx][:, None].repeat(1, 7)
                    * cluster_boxes[max_idx],
                    dim=0,
                ) / torch.sum(cluster_scores[max_idx])

        for k in range(len(fused_scores)):
            fused_scores[k] = (
                torch.mean(cluster_scores[k])
                * min(n_models, len(cluster_scores[k]))
                / n_models
            )
            fused_labels[k], _ = torch.mode(cluster_labels[k])

        fused_boxes = torch.stack(fused_boxes)
        fused_scores = torch.stack(fused_scores)
        fused_labels = torch.stack(fused_labels)

        return fused_boxes, fused_scores, fused_labels


def get_voxel_centers(voxel_coords, downsample_times, voxel_size, point_cloud_range):
    """
    Args:
        voxel_coords: (N, 3)
        downsample_times:
        voxel_size:
        point_cloud_range:

    Returns:

    """
    assert voxel_coords.shape[1] == 3
    voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
    voxel_size = (
        torch.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times
    )
    pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    return voxel_centers


def scatter_point_inds(indices, point_inds, shape):
    ret = -1 * torch.ones(*shape, dtype=point_inds.dtype, device=point_inds.device)
    ndim = indices.shape[-1]
    flattened_indices = indices.view(-1, ndim)
    slices = [flattened_indices[:, i] for i in range(ndim)]
    ret[slices] = point_inds
    return ret


def generate_voxel2pinds(sparse_tensor):
    device = sparse_tensor.indices.device
    batch_size = sparse_tensor.batch_size
    spatial_shape = sparse_tensor.spatial_shape
    indices = sparse_tensor.indices.long()
    point_indices = torch.arange(indices.shape[0], device=device, dtype=torch.int32)
    output_shape = [batch_size] + list(spatial_shape)
    v2pinds_tensor = scatter_point_inds(indices, point_indices, output_shape)
    return v2pinds_tensor
