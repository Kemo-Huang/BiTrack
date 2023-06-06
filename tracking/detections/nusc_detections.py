from typing import Dict, List

import numpy as np
from nuscenes.eval.common.loaders import load_prediction
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.tracking.data_classes import TrackingBox
from pyquaternion import Quaternion

from .detections import Detections


def get_detection_data_all(json_file_path):
    print('Loading detections')
    eval_boxes, _ = load_prediction(json_file_path, 10000, DetectionBox)
    det_boxes_dict: Dict[str, List[DetectionBox]] = eval_boxes.boxes
    trk_boxes_dict = {k: [TrackingBox(
        b.sample_token, b.translation, b.size, b.rotation, b.velocity, b.ego_translation, b.num_pts, '', b.detection_name, b.detection_score
    ) for b in v if b.detection_name == 'car'] for k, v in det_boxes_dict.items()}
    return trk_boxes_dict

def get_detection_data(objs_dict: Dict[str, List[TrackingBox]], sample_token):
    objs = objs_dict[sample_token]
    boxes = [np.array([*obj.translation, *obj.size, Quaternion(*obj.rotation).radians]) for obj in objs]
    return Detections(boxes, objs, None, None, None)