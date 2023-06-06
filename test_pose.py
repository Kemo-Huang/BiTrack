from matplotlib import pyplot as plt

from utils import (Calibration, get_global_boxes_from_lidar,
                   get_lidar_boxes_from_objs, get_objects_from_label,
                   get_poses_from_file)


def main():
    tracks = get_objects_from_label('output/simple/data/0001.txt', track=True)
    tracks = [trk for trk in tracks if trk.cls_type != 'DontCare']
    poses = get_poses_from_file('data/tracking/oxts/0001.txt')
    calib = Calibration('data/tracking/calib/0001.txt')
    frame_tracks_dict = {}
    for trk in tracks:
        if trk.sample_id not in frame_tracks_dict:
            frame_tracks_dict[trk.sample_id] = [trk]
        else:
            frame_tracks_dict[trk.sample_id].append(trk)
    plt.figure()
    for frame_id in range(5, 15):
        boxes = get_lidar_boxes_from_objs(frame_tracks_dict[frame_id], calib)
        plt.scatter(-boxes[:, 1], boxes[:, 0])
    plt.figure()
    for frame_id in range(5, 15):
        boxes = get_lidar_boxes_from_objs(frame_tracks_dict[frame_id], calib)
        boxes = get_global_boxes_from_lidar(boxes, poses[frame_id])
        plt.scatter(-boxes[:, 1], boxes[:, 0])
    plt.show()

main()
        