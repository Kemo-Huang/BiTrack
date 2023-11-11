from nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import DetectionEval


def main():
    nusc = NuScenes("v1.0-trainval", dataroot="data/nuscenes", verbose=True)
    nusc_eval = DetectionEval(
        nusc,
        config=config_factory("detection_cvpr_2019"),
        result_path="data/nuscenes/my_bevdet_results.json",
        eval_set="val",
        output_dir="output/nuscenes_det/bevdet",
        verbose=True,
    )
    nusc_eval.main(render_curves=False)


if __name__ == "__main__":
    main()
