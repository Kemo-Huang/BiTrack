import numpy as np


class Detections:
    def __init__(self, boxes3d: np.ndarray, objs, similarity, embeds, coor_2d_inds):
        self.boxes3d = boxes3d
        self.objs = objs
        self.similarity = similarity
        self.embeds = embeds
        self.corr_2d_inds = coor_2d_inds
        if boxes3d is not None:
            assert len(boxes3d) == len(objs)
        if similarity is not None:
            assert len(similarity) == len(boxes3d)
        if embeds is not None:
            assert len(embeds) == len(coor_2d_inds)

    def __len__(self):
        if self.boxes3d is not None:
            return len(self.boxes3d)
        else:
            return len(self.corr_2d_inds)

    def append_3d(self, b):
        self.boxes3d = np.concatenate((self.boxes3d, b.boxes3d))
        a_objs = self.objs if isinstance(self.objs, np.ndarray) else np.array(self.objs)
        b_objs = b.objs if isinstance(b.objs, np.ndarray) else np.array(b.objs)
        self.objs = np.concatenate((a_objs, b_objs))

        assert len(self.boxes3d) == len(self.objs)

        if self.similarity is not None and b.similarity is not None:
            self.similarity = np.concatenate((self.similarity, b.similarity))
        if self.embeds is not None and b.embeds is not None:
            self.embeds = np.concatenate((self.embeds, b.embeds))
        if self.corr_2d_inds is not None and b.corr_2d_inds is not None:
            self.corr_2d_inds = np.concatenate((self.corr_2d_inds, b.corr_2d_inds))

    def delete_2d(self, b):
        remain_inds = np.setdiff1d(self.corr_2d_inds, b.corr_2d_inds)
        remain_mask = np.isin(self.corr_2d_inds, remain_inds)
        self.corr_2d_inds = remain_inds
        if self.embeds is not None:
            self.embeds = self.embeds[remain_mask]
