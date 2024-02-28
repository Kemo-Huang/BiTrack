import json
from argparse import ArgumentParser

from nuscenes.eval.common.config import config_factory
from nuscenes.eval.tracking.evaluate import TrackingEval

TRK_CLASSES = [
    "bicycle",
    "motorcycle",
    "pedestrian",
    "bus",
    "car",
    "trailer",
    "truck",
]


def main():
    parser = ArgumentParser()
    parser.add_argument("split", type=str)
    parser.add_argument("tag", type=str)
    args = parser.parse_args()

    with open(f"output/nuscenes/{args.split}/{args.tag}/results.json", "r") as f:
        data = json.load(f)
    results = data["results"]
    for sample_token, objs in results.items():
        for obj in objs:
            if "detection_name" in obj:
                obj["tracking_name"] = obj["detection_name"]
            if "detection_score" in obj:
                obj["tracking_score"] = obj["detection_score"]
        results[sample_token] = [
            obj for obj in objs if obj["tracking_name"] in TRK_CLASSES
        ]
    data["results"] = results
    with open(
        f"output/nuscenes/{args.split}/{args.tag}/tracking_results.json", "w"
    ) as f:
        json.dump(data, f)

    track_eval = TrackingEval(
        config=config_factory("tracking_nips_2019"),
        result_path=f"output/nuscenes/{args.split}/{args.tag}/tracking_results.json",
        eval_set=args.split,
        output_dir=f"output/nuscenes/{args.split}/{args.tag}",
        nusc_version="v1.0-trainval",
        nusc_dataroot="data/nuscenes",
    )
    track_eval.main(render_curves=False)


if __name__ == "__main__":
    main()
