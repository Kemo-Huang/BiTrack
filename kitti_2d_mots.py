from argparse import ArgumentParser
from configparser import ConfigParser
from pathlib import Path

import numpy as np
import torch
import tqdm
from PIL import Image
from torchvision.transforms import functional as F

from segmentation.point_track.point_track import EmbedNet
from segmentation.spatial_embeddings.spatial_embeddings import BranchedERFNet
from tracking.mots_util import TrackHelper


def main():
    parser = ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("split", type=str)
    parser.add_argument("--load_seg", action="store_true")
    parser.add_argument("--load_emb", action="store_true")
    parser.add_argument("--embedding", action="store_true")
    parser.add_argument("--tracking", action="store_true")

    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config)
    root_dir = Path(config["data"]["root_dir"]) / args.split
    data_dir = root_dir / "image_02"
    seg_out_dir = root_dir / "seg_out" / config["detection"]["seg_name"]
    if args.embedding:
        emb_out_dir = root_dir / "seg_emb_out" / config["detection"]["seg_emb_name"]

    # clustering config
    min_pixel = 160
    threshold = 0.94
    n_sigma = 2
    # embedding test config
    num_points = 1000

    if not args.load_seg:
        seg_net = BranchedERFNet(num_classes=[2 + n_sigma, 1], input_channel=3)
        seg_net.load_state_dict(torch.load(config["detection"]["seg_ckpt"]))
        seg_net.cuda()
        seg_net.eval()

    if args.embedding:
        embed_net = EmbedNet(
            num_points=num_points, border_ic=3, env_points=500, outputD=32, margin=0.2
        )
        if not args.load_emb:
            embed_net.load_state_dict(torch.load(config["detection"]["seg_emb_ckpt"]))
            embed_net.cuda()
            embed_net.eval()
            embed_net.category_embedding = embed_net.category_embedding.cuda()

        if args.tracking:
            track_out_dir = Path(f"output/kitti/{args.split}/point_track/data")
            track_out_dir.mkdir(parents=True, exist_ok=True)
            trackHelper = TrackHelper(
                track_out_dir, alive_car=30, car=True, mask_iou=False
            )

    seq_paths = data_dir.iterdir()

    for seq in seq_paths:
        seq_seg_out_dir = seg_out_dir / seq.name
        seq_seg_out_dir.mkdir(exist_ok=True, parents=True)

        if args.embedding:
            seq_emb_out_dir = emb_out_dir / seq.name
            seq_emb_out_dir.mkdir(exist_ok=True, parents=True)

        pbar = tqdm.tqdm(list(seq.iterdir()))
        pbar.set_description(seq.name)
        for img_file in pbar:
            img = Image.open(img_file)
            w, h = img.size
            img = F.pil_to_tensor(img).float() / 255
            seg_filename = seq_seg_out_dir / f"{img_file.stem}.png"

            if args.load_seg:
                instance_map = Image.open(seg_filename)
                instance_map = F.pil_to_tensor(instance_map)
            else:
                img = img.cuda()
                seg_input = F.pad(img, (0, 0, 1248 - w, 384 - h), padding_mode="edge")
                seg_input = seg_input[None]  # batch size = 1
                with torch.no_grad():
                    out = seg_net.forward(seg_input)[0]
                out = out[:, :h, :w]
                instance_map = seg_net.cluster(
                    out, threshold=threshold, min_pixel=min_pixel, n_sigma=n_sigma
                )
                # instance_map (H, W)
                Image.fromarray(instance_map.cpu().numpy()).save(seg_filename)

            if args.embedding:
                fg_points, bg_points, xyxys, masks = embed_net.preprocessing(
                    img, instance_map, cuda=not args.load_seg
                )
                if len(xyxys) > 0:
                    masks = masks.cpu().numpy()

                    if args.load_emb:
                        embeds = np.load(seq_emb_out_dir / f"{img_file.stem}.npy")
                    else:
                        with torch.no_grad():
                            embeds = embed_net.forward(fg_points, bg_points, xyxys)
                        embeds = embeds.cpu().numpy()
                        np.save(seq_emb_out_dir / f"{img_file.stem}.npy", embeds)
                else:
                    embeds = np.array([])
                    masks = np.array([])

                if args.tracking:
                    trackHelper.tracking(seq.name, int(img_file.stem), embeds, masks)

    if args.tracking:
        trackHelper.export_last_video()


if __name__ == "__main__":
    main()
