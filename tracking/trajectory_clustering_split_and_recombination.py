from typing import Dict, List, Tuple

import numpy as np

from utils import KittiTrack3d, visualize_trajectories


def get_overlaps_of_trajectories(
    a_trajectories: Dict[int, Tuple[List[np.ndarray], List[KittiTrack3d]]],
    b_trajectories: Dict[int, Tuple[List[np.ndarray], List[KittiTrack3d]]],
) -> Tuple[List[int], List[int], List[List[Tuple[int, int]]]]:
    """
    Trajectories should be ordered by frame ids.

    Returns:
        overlap_pair_inds: N x M
    """
    n = len(a_trajectories)
    m = len(b_trajectories)
    overlap_pair_inds = [[[] for _ in range(m)] for _ in range(n)]
    a_ids = list(a_trajectories.keys())
    b_ids = list(b_trajectories.keys())
    for i, a_id in enumerate(a_ids):
        _, a_objs = a_trajectories[a_id]
        a_frames = [a_obj.sample_id for a_obj in a_objs]
        # check frames are sorted
        assert all(
            a_frames[idx] < a_frames[idx + 1] for idx in range(len(a_frames) - 1)
        ), str(a_frames)

        for j, b_id in enumerate(b_ids):
            _, b_objs = b_trajectories[b_id]
            b_frames = [b_obj.sample_id for b_obj in b_objs]
            # check frames are sorted
            assert all(
                b_frames[idx] < b_frames[idx + 1] for idx in range(len(b_frames) - 1)
            ), str(b_frames)

            # two-pointers search
            a_i = 0
            b_j = 0
            while a_i < len(a_frames) and b_j < len(b_frames):
                if a_frames[a_i] == b_frames[b_j]:
                    if np.array_equal(a_objs[a_i].loc, b_objs[b_j].loc):
                        overlap_pair_inds[i][j].append((a_i, b_j))
                    a_i += 1
                    b_j += 1
                else:
                    if a_frames[a_i] > b_frames[b_j]:
                        b_j += 1
                    else:
                        a_i += 1
    return a_ids, b_ids, overlap_pair_inds


def get_clusters(overlap_arr) -> List[Tuple[List[int]]]:
    n = len(overlap_arr)
    m = len(overlap_arr[0])
    a_idx_set = set(range(n))
    b_idx_set = set(range(m))
    clusters = []
    while a_idx_set or b_idx_set:
        cluster_a = []
        cluster_b = []
        if a_idx_set:
            cur_idx = a_idx_set.pop()
            cluster_a.append(cur_idx)
            cur_inds = [
                j for j in range(m) if overlap_arr[cur_idx][j] and j in b_idx_set
            ]
            is_a = True
        else:
            cur_idx = b_idx_set.pop()
            cluster_b.append(cur_idx)
            cur_inds = [
                i for i in range(n) if overlap_arr[i][cur_idx] and i in a_idx_set
            ]
            is_a = False
        while len(cur_inds) > 0:
            next_inds = []
            for x in cur_inds:
                if is_a:
                    b_idx_set.remove(x)
                    cluster_b.append(x)
                    next_inds += [
                        i for i in range(n) if overlap_arr[i][x] and i in a_idx_set
                    ]
                else:
                    a_idx_set.remove(x)
                    cluster_a.append(x)
                    next_inds += [
                        j for j in range(m) if overlap_arr[x][j] and j in b_idx_set
                    ]
            cur_inds = set(next_inds)
            is_a = not is_a
        clusters.append((cluster_a, cluster_b))
    return clusters


def group_consecutive_inds_dict(inds_dict: Dict[int, List[int]]):
    # this is inplace modification
    for key, indices in inds_dict.items():
        consecutive_inds = []
        last_idx = None
        for x in indices:
            if last_idx is None:
                last_idx = x
                consecutive_inds.append([x])
            else:
                if x == last_idx + 1:
                    consecutive_inds[-1].append(x)
                else:
                    consecutive_inds.append([x])
        inds_dict[key] = consecutive_inds


def merge_distinct(
    common_inds_dict,
    ids,
    trajectories,
    cluster_distinct_inds,
    all_boxes,
    all_objs,
    all_frames,
):
    for idx, common_inds_list in common_inds_dict.items():
        boxes, objs = trajectories[ids[idx]]
        distinct_inds_list = cluster_distinct_inds[idx]
        valid_mask = [False] * len(distinct_inds_list)
        common_inds_list.sort()
        i = 0
        j = 0
        while i < len(common_inds_list) and j < len(distinct_inds_list):
            if distinct_inds_list[j][0] == 0:
                if common_inds_list[i][0] == distinct_inds_list[j][-1] + 1:
                    valid_mask[j] = True
                    i += 1
                    j += 1
                elif common_inds_list[i][0] < distinct_inds_list[j][-1] + 1:
                    i += 1
                else:
                    j += 1
            else:
                if common_inds_list[i][-1] == distinct_inds_list[j][0] - 1:
                    valid_mask[j] = True
                    i += 1
                    j += 1
                elif common_inds_list[i][-1] < distinct_inds_list[j][0] - 1:
                    i += 1
                else:
                    j += 1

        distinct_inds_list_new = []
        for x in range(len(distinct_inds_list)):
            if valid_mask[x]:
                for distinct_idx in distinct_inds_list[x]:
                    if objs[distinct_idx].sample_id not in all_frames:
                        all_boxes.append(boxes[distinct_idx])
                        all_objs.append(objs[distinct_idx])
                        all_frames.append(objs[distinct_idx].sample_id)
            else:
                distinct_inds_list_new.append(distinct_inds_list[x])
        cluster_distinct_inds[idx] = distinct_inds_list_new


def use_forward_common_trajectories(
    a_trajectories: Dict[int, Tuple[List[np.ndarray], List[KittiTrack3d]]],
    b_trajectories: Dict[int, Tuple[List[np.ndarray], List[KittiTrack3d]]],
):
    if len(a_trajectories) == 0 or len(b_trajectories) == 0:
        return a_trajectories if len(a_trajectories) > 0 else b_trajectories
    a_ids, b_ids, overlap_pair_inds = get_overlaps_of_trajectories(
        a_trajectories, b_trajectories
    )
    for i, a_id in enumerate(a_ids):
        boxes, objs = a_trajectories[a_id]
        inds = [[pair[0] for pair in pair_inds] for pair_inds in overlap_pair_inds[i]]
        inds = sum(inds, [])
        boxes = [boxes[x] for x in inds]
        objs = [objs[x] for x in inds]
        a_trajectories[a_id] = (boxes, objs)
    return a_trajectories


def merge_forward_backward_trajectories(
    a_trajectories: Dict[int, Tuple[List[np.ndarray], List[KittiTrack3d]]],
    b_trajectories: Dict[int, Tuple[List[np.ndarray], List[KittiTrack3d]]],
    visualize_contradictions=False,
    merge_common_only=True,
):
    if len(a_trajectories) == 0 or len(b_trajectories) == 0:
        return a_trajectories if len(a_trajectories) > 0 else b_trajectories
    a_ids, b_ids, overlap_pair_inds = get_overlaps_of_trajectories(
        a_trajectories, b_trajectories
    )
    is_overlapped = [
        [len(overlap_pair_inds[i][j]) > 0 for j in range(len(b_ids))]
        for i in range(len(a_ids))
    ]
    clusters = get_clusters(is_overlapped)
    # merge
    merged_trajectories = {}
    cur_id = 1
    for cluster in clusters:
        cluster_a_inds: List[int] = cluster[0]
        cluster_b_inds: List[int] = cluster[1]

        cluster_common_objs = []
        cluster_common_boxes = []
        cluster_a_common_inds = {x: [] for x in cluster_a_inds}
        cluster_b_common_inds = {x: [] for x in cluster_b_inds}
        for a_idx in cluster_a_inds:
            a_boxes, a_objs = a_trajectories[a_ids[a_idx]]
            for b_idx in cluster_b_inds:
                for a_common_idx, b_common_idx in overlap_pair_inds[a_idx][b_idx]:
                    cluster_common_objs.append(a_objs[a_common_idx])
                    cluster_common_boxes.append(a_boxes[a_common_idx])
                    cluster_a_common_inds[a_idx].append(a_common_idx)
                    cluster_b_common_inds[b_idx].append(b_common_idx)

        if merge_common_only:
            cluster_common_frames = [x.sample_id for x in cluster_common_objs]
            if len(set(cluster_common_frames)) == len(cluster_common_frames):
                if len(cluster_common_objs) > 0:
                    for x in cluster_common_objs:
                        x.tracking_id = cur_id
                    merged_trajectories[cur_id] = (
                        cluster_common_boxes,
                        cluster_common_objs,
                    )
                    cur_id += 1
                continue

        else:
            cluster_a_distinct_boxes = []
            cluster_a_distinct_objs = []
            cluster_b_distinct_boxes = []
            cluster_b_distinct_objs = []
            for a_idx in cluster_a_inds:
                a_boxes, a_objs = a_trajectories[a_ids[a_idx]]
                a_distinct_inds = [
                    x
                    for x in range(len(a_objs))
                    if x not in cluster_a_common_inds[a_idx]
                ]
                cluster_a_distinct_boxes = [a_boxes[x] for x in a_distinct_inds]
                cluster_a_distinct_objs = [a_objs[x] for x in a_distinct_inds]
            for b_idx in cluster_b_inds:
                b_boxes, b_objs = b_trajectories[b_ids[b_idx]]
                b_distinct_inds = [
                    x
                    for x in range(len(b_objs))
                    if x not in cluster_b_common_inds[b_idx]
                ]
                cluster_b_distinct_boxes = [b_boxes[x] for x in b_distinct_inds]
                cluster_b_distinct_objs = [b_objs[x] for x in b_distinct_inds]

            cluster_all_boxes = (
                cluster_common_boxes
                + cluster_a_distinct_boxes
                + cluster_b_distinct_boxes
            )
            cluster_all_objs = (
                cluster_common_objs + cluster_a_distinct_objs + cluster_b_distinct_objs
            )
            cluster_all_frames = [x.sample_id for x in cluster_all_objs]

            if len(set(cluster_all_frames)) == len(cluster_all_frames):
                if len(cluster_all_objs) > 0:
                    for x in cluster_all_objs:
                        x.tracking_id = cur_id
                    merged_trajectories[cur_id] = (cluster_all_boxes, cluster_all_objs)
                    cur_id += 1
                continue

        if visualize_contradictions:
            cluster_a_boxes = []
            for a_idx in cluster_a_inds:
                a_boxes, _ = a_trajectories[a_ids[a_idx]]
                cluster_a_boxes.append(a_boxes)
            visualize_trajectories(cluster_a_boxes)
            cluster_b_boxes = []
            for b_idx in cluster_b_inds:
                b_boxes, _ = b_trajectories[b_ids[b_idx]]
                cluster_b_boxes.append(b_boxes[::-1])  # reverse direction
            visualize_trajectories(cluster_b_boxes)

        # (a_idx, b_idx, a_last_idx, b_last_idx, len, [frames])
        common_tracklets = []
        for a_idx in cluster_a_inds:
            for b_idx in cluster_b_inds:
                last_a_idx = None
                last_b_idx = None
                cur_tracklet = []
                b_boxes, b_objs = b_trajectories[b_ids[b_idx]]
                b_len = len(b_boxes)
                for a_common_idx, b_common_idx in overlap_pair_inds[a_idx][b_idx]:
                    cur_frame = b_objs[b_common_idx].sample_id
                    b_common_idx = (
                        b_len - 1 - b_common_idx
                    )  # reverse indices for backward scores
                    if last_a_idx is None:
                        cur_tracklet = [
                            a_idx,
                            b_idx,
                            a_common_idx,
                            b_common_idx,
                            1,
                            [cur_frame],
                        ]
                    else:
                        if (
                            a_common_idx == last_a_idx + 1
                            and b_common_idx == last_b_idx - 1
                        ):
                            # consecutive
                            cur_tracklet[2] = a_common_idx
                            cur_tracklet[4] += 1
                            cur_tracklet[5].append(cur_frame)
                        else:
                            # not consecutive
                            common_tracklets.append(cur_tracklet)
                            cur_tracklet = [
                                a_idx,
                                b_idx,
                                a_common_idx,
                                b_common_idx,
                                1,
                                [cur_frame],
                            ]
                    last_a_idx = a_common_idx
                    last_b_idx = b_common_idx
                if len(cur_tracklet) > 0:
                    common_tracklets.append(cur_tracklet)

        # sort tracklets by last inds
        a_sorted_tracklets = {a_idx: [] for a_idx in cluster_a_inds}
        b_sorted_tracklets = {b_idx: [] for b_idx in cluster_b_inds}
        for i, tracklet in enumerate(common_tracklets):
            a_idx, b_idx, a_last_idx, b_last_idx, _, _ = tracklet
            a_sorted_tracklets[a_idx].append((a_last_idx, i))
            b_sorted_tracklets[b_idx].append((b_last_idx, i))
        for x in a_sorted_tracklets.values():
            x.sort()
        for x in b_sorted_tracklets.values():
            x.sort()
        a_sorted_tracklets = {
            k: [x[1] for x in v] for k, v in a_sorted_tracklets.items()
        }
        b_sorted_tracklets = {
            k: [x[1] for x in v] for k, v in b_sorted_tracklets.items()
        }

        before_scores = [-1] * len(common_tracklets)  # forward
        after_scores = [-1] * len(common_tracklets)  # forward
        before_choices = [-1] * len(common_tracklets)
        after_choices = [-1] * len(common_tracklets)

        before_bad_scores = [-1] * len(common_tracklets)
        after_bad_scores = [-1] * len(common_tracklets)
        before_bad_choices = [-1] * len(common_tracklets)
        after_bad_choices = [-1] * len(common_tracklets)

        for i, tracklet in enumerate(common_tracklets):
            a_idx, b_idx, a_last_idx, b_last_idx, _, _ = tracklet
            cur_a_tracklets: list = a_sorted_tracklets[a_idx]
            # reverse order
            cur_b_tracklets: list = b_sorted_tracklets[b_idx]
            a_k = cur_a_tracklets.index(i)
            b_k = cur_b_tracklets.index(i)
            if a_k == 0:
                a_before_score = -1
                a_before_tracklet_idx = "placeholder"
            else:
                a_before_tracklet_idx = cur_a_tracklets[a_k - 1]
                a_before_score = common_tracklets[a_before_tracklet_idx][2]

            b_before_score = b_last_idx
            if b_k < len(cur_b_tracklets) - 1:
                b_before_tracklet_idx = cur_b_tracklets[b_k + 1]
            else:
                b_before_tracklet_idx = "placeholder"

            a_after_score = a_last_idx
            if a_k < len(cur_a_tracklets) - 1:
                a_after_tracklet_idx = cur_a_tracklets[a_k + 1]
            else:
                a_after_tracklet_idx = "placeholder"

            if b_k == 0:
                b_after_score = -1
                b_after_tracklet_idx = "placeholder"
            else:
                b_after_tracklet_idx = cur_b_tracklets[b_k - 1]
                b_after_score = common_tracklets[b_after_tracklet_idx][3]

            # ========================================================

            if a_before_score >= b_before_score:
                before_scores[i] = a_before_score
                before_choices[i] = a_before_tracklet_idx
                before_bad_scores[i] = b_before_score
                before_bad_choices[i] = b_before_tracklet_idx
            else:
                before_scores[i] = b_before_score
                before_choices[i] = b_before_tracklet_idx
                before_bad_scores[i] = a_before_score
                before_bad_choices[i] = a_before_tracklet_idx

            if a_after_score >= b_after_score:
                after_scores[i] = a_after_score
                after_choices[i] = a_after_tracklet_idx
                after_bad_scores[i] = b_after_score
                after_bad_choices[i] = b_after_tracklet_idx
            else:
                after_scores[i] = b_after_score
                after_choices[i] = b_after_tracklet_idx
                after_bad_scores[i] = a_after_score
                after_bad_choices[i] = a_after_tracklet_idx

        all_scores = before_scores + before_bad_scores + after_scores + after_bad_scores
        all_choices = (
            before_choices + before_bad_choices + after_choices + after_bad_choices
        )
        connections = [
            [None, None] for _ in range(len(common_tracklets))
        ]  # (before, after) forward
        tracklets_frames = [set(x[5]) for x in common_tracklets]
        for x in np.argsort(all_scores)[::-1]:
            cur_choice = all_choices[x]
            if cur_choice == "placeholder":
                if x < 2 * len(common_tracklets):  # before
                    x %= len(common_tracklets)
                    if connections[x][0] is None:
                        connections[x][0] = "placeholder"
                else:  # after
                    x %= len(common_tracklets)
                    if connections[x][1] is None:
                        connections[x][1] = "placeholder"
                continue
            if x < 2 * len(common_tracklets):  # before
                x %= len(common_tracklets)
                if (
                    connections[x][0] is None
                    and connections[cur_choice][1] is None
                    and tracklets_frames[cur_choice].isdisjoint(tracklets_frames[x])
                ):
                    connections[x][0] = cur_choice
                    connections[cur_choice][1] = x
                    union = tracklets_frames[x].union(tracklets_frames[cur_choice])
                    tracklets_frames[x] = union
                    tracklets_frames[cur_choice] = union
            else:  # after
                x %= len(common_tracklets)
                if (
                    connections[x][1] is None
                    and connections[cur_choice][0] is None
                    and tracklets_frames[cur_choice].isdisjoint(tracklets_frames[x])
                ):
                    connections[x][1] = cur_choice
                    connections[cur_choice][0] = x
                    union = tracklets_frames[x].union(tracklets_frames[cur_choice])
                    tracklets_frames[x] = union
                    tracklets_frames[cur_choice] = union

        merged_tracklet_inds_list = []
        visited = [False] * len(connections)
        for i, connection in enumerate(connections):
            if visited[i]:
                continue
            if connection[0] == "placeholder" or connection[0] is None:
                cur_tracklet = [i]
                visited[i] = True
                next_i = connection[1]
                while next_i != "placeholder" and next_i is not None:
                    cur_tracklet.append(next_i)
                    visited[next_i] = True
                    next_i = connections[next_i][1]
                merged_tracklet_inds_list.append(cur_tracklet)

        cur_merged_trajectories = {}
        for merged_tracklet_inds in merged_tracklet_inds_list:
            all_boxes = []
            all_objs = []

            for idx in merged_tracklet_inds:
                (
                    a_idx,
                    b_idx,
                    a_last_idx,
                    b_last_idx,
                    track_len,
                    cur_frames,
                ) = common_tracklets[idx]
                a_boxes, a_objs = a_trajectories[a_ids[a_idx]]
                b_boxes, b_objs = b_trajectories[b_ids[b_idx]]
                a_first_idx = a_last_idx - track_len + 1
                b_first_idx = len(b_boxes) - b_last_idx - 1  # reverse back
                b_last_idx = b_first_idx + track_len - 1

                # a, b are the same
                all_boxes += a_boxes[a_first_idx : a_last_idx + 1]
                all_objs += a_objs[a_first_idx : a_last_idx + 1]

            for x in all_objs:
                x.tracking_id = cur_id
            cur_merged_trajectories[cur_id] = (all_boxes, all_objs)
            cur_id += 1

        if visualize_contradictions:
            visualize_trajectories([x[0] for x in cur_merged_trajectories.values()])

        merged_trajectories.update(cur_merged_trajectories)

    return merged_trajectories
