from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE

from utils import Calibration, get_instance_ids_of_boxes_hungarian


def main():
    min_n_pts = 3

    all_embeds = []
    corres_gt_ids = []

    for i in range(21):
        seq = str(i).zfill(4)
        embed_dir = Path(f"data/tracking/embeddings/{seq}")
        instance_dir = Path(f"data/tracking/seg_instances/{seq}")
        lidar_dir = Path(f"data/tracking/cropped_points/gt/{seq}")
        calib = Calibration(f"data/tracking/calib/{seq}.txt")

        for instance_file in sorted(instance_dir.iterdir()):
            embed_file = embed_dir / f"{instance_file.stem}.npy"
            if embed_file.exists():
                cur_lidar_dir = lidar_dir / instance_file.stem
                if cur_lidar_dir.exists():
                    gt_ids = []
                    inside_points = []
                    for lidar_file in cur_lidar_dir.iterdir():
                        gt_ids.append(int(lidar_file.stem))
                        inside_points.append(
                            np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)[
                                :, :3
                            ]
                        )
                    instance_img = np.array(Image.open(instance_file))
                    instance_ids, _ = get_instance_ids_of_boxes_hungarian(
                        instance_img, inside_points, calib, min_n_pts
                    )
                    embeds = np.load(embed_file)

                    good_box_mask = instance_ids > 0
                    all_embeds.append(embeds[instance_ids[good_box_mask] - 1])
                    corres_gt_ids.append(
                        np.array(gt_ids, dtype=int)[good_box_mask] + 1000 * i
                    )

    all_embeds = np.concatenate(all_embeds)
    corres_gt_ids = np.concatenate(corres_gt_ids)
    print(all_embeds.shape, corres_gt_ids.shape)

    tsne = TSNE(n_components=2, init="pca", learning_rate="auto")
    X_new = tsne.fit_transform(all_embeds)
    plt.scatter(X_new[:, 0], X_new[:, 1], c=corres_gt_ids)
    plt.show()


main()
