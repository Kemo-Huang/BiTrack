from nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import DetectionEval


def main():
    nusc = NuScenes("v1.0-mini", dataroot="data/nuscenes_mini", verbose=False)
    # nusc = NuScenes("v1.0-trainval", dataroot="data/nuscenes", verbose=True)
    nusc_eval = DetectionEval(
        nusc,
        config=config_factory("detection_cvpr_2019"),
        # result_path="data/nuscenes/results_nusc_val.json",
        # result_path="data/nuscenes/mini_val_results.json",
        result_path="output/nuscenes/mini/mini/results.json",
        eval_set="mini_val",
        # eval_set="val",
        output_dir="output/nuscenes_det/det",
        # output_dir="output/nuscenes_det/val",
        verbose=True,
    )
    nusc_eval.main(render_curves=False)


if __name__ == "__main__":
    main()
