from argparse import ArgumentParser

from nuscenes.eval.common.config import config_factory
from nuscenes.eval.tracking.evaluate import TrackingEval


def main():
    parser = ArgumentParser()
    parser.add_argument("tag", type=str)
    args = parser.parse_args()
    track_eval = TrackingEval(
        config=config_factory("tracking_nips_2019"),
        result_path=f"output/nusc/{args.tag}/results.json",
        eval_set="val",
        output_dir=f"output/nusc/{args.tag}",
        nusc_version="v1.0-trainval",
        nusc_dataroot="data/nuscenes",
    )
    track_eval.main(render_curves=False)


if __name__ == "__main__":
    main()
