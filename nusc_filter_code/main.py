from nuscenes.nuscenes import NuScenes
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud
from scipy.spatial import Delaunay
import scipy
import os.path as osp
import pickle
from tqdm import tqdm
import os


def get_all_tokens_by_scene(nusc, scene_idx):
    scene = nusc.scene[scene_idx]
    first_sampling_token = scene["first_sample_token"]
    tokens = [first_sampling_token]
    token = first_sampling_token
    sample = nusc.get('sample', token)
    while sample['next'] != '':
        token = sample['next']
        tokens.append(token)
        sample = nusc.get('sample', token)

    return tokens


def corners(box_size, box_center, box_rotation) -> np.ndarray:
    w, l, h = box_size * 1.0
    # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
    x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    corners = np.vstack((x_corners, y_corners, z_corners))
    # Rotate
    corners = np.dot(box_rotation.rotation_matrix, corners)
    # Translate
    x, y, z = box_center
    corners[0, :] = corners[0, :] + x
    corners[1, :] = corners[1, :] + y
    corners[2, :] = corners[2, :] + z
    return corners


def get_qt_from_sample_token(nusc, sample_token):
    sample_record = nusc.get('sample', sample_token)
    # get box in the sample data
    lidar_data_token = sample_record['data']['LIDAR_TOP']
    lidar_data = nusc.get("sample_data", lidar_data_token)

    calib_data = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    ego_data = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    pc_raw, pc_dim4 = read_bin_velodyne(lidar_data["filename"])

    return calib_data, ego_data, pc_raw, pc_dim4, lidar_data


def trans_box_to_lidar(nusc, calib_data, ego_data, anno_data_token=None, box_size=None, box_center=None,
                       box_rotation=None):
    if anno_data_token is not None:
        anno_data = nusc.get("sample_annotation", anno_data_token)
    elif box_size is not None:
        # 从global frame转换到ego vehicle frame
        quaternion = Quaternion(ego_data['rotation']).inverse
        box_center -= np.array(ego_data['translation'])
        center = np.dot(quaternion.rotation_matrix, box_center)
        orientation = quaternion * box_rotation
        # 从ego vehicle frame转换到sensor frame
        quaternion = Quaternion(calib_data['rotation']).inverse
        center -= np.array(calib_data['translation'])
        center = np.dot(quaternion.rotation_matrix, center)
        orientation = quaternion * orientation
        return center, orientation


def get_points_in_box(raw_pc, box_size, box_translation, box_rotation):
    """

    :param raw_pc: ndarray(N,3)
    :param box_size: ndarray(3, )
    :param box_translation: ndarray(3,)
    :param box_rotation: Quaternion (4,)
    :return: flag (N, )
    """
    hull = corners(box_size, box_translation, box_rotation)
    hull = hull.T
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        flag = hull.find_simplex(raw_pc) >= 0
    except scipy.spatial.qhull.QhullError:
        print('Warning: not a hull %s' % str(hull))
        flag = np.zeros(raw_pc.shape[0], dtype=np.bool)

    return flag


def read_bin_velodyne(path):
    lidar_points_path = osp.join("/home/leicy/data/nuscenes/v1.0-trainval/", path)
    pc = LidarPointCloud.from_file(lidar_points_path)
    return pc.points[:3, :].T, pc.points.T


def save_pc_bin(pc_dim4, flag, root_dir, filename):
    pc_save = pc_dim4[flag]
    pc_save = np.hstack((pc_save, np.zeros((pc_save.shape[0], 1))))
    pc_save = pc_save.reshape(-1)
    # save to .bin
    pc_save.astype(np.float32).tofile(osp.join(root_dir, filename))



if __name__ == "__main__":

    nuscene_dir = "/home/leicy/data/nuscenes/v1.0-trainval"
    version_type = "v1.0-trainval"

    result_file_dir = "./result_dir"
    point_cloud_save_dir = "./filtered_pc"

    nusc = NuScenes(version=version_type, dataroot=nuscene_dir, verbose=True)

    # car_val: 只考虑car的val集合，all_val: 所有类的val集合，car_test: 只考虑car的test集合，all_test: 所有类的test集合
    # 运行car_test和all_test的时候需要更改一下nuscene_dir 和 version_type
    # 运行后点云保存在./filtered_pc/下
    # val:6019帧 test: 6008帧
    result_file_type = ["car_val", "all_val", "car_test", "all_test"]
    print(f"#############{result_file_type}#################")
    for file_type in result_file_type:

        save_file_dir = osp.join(point_cloud_save_dir, file_type)
        if not osp.exists(save_file_dir):
            os.makedirs(save_file_dir)

        filter_result = pickle.load(open(osp.join(result_file_dir, file_type+"_result_nusc_filter.pkl"), "rb"))


        for key, value in tqdm(filter_result.items()):
            sample_token = key
            calib_data, ego_data, pc_raw, pc_dim5, lidar_data = get_qt_from_sample_token(nusc, sample_token)
            # create flag as the same size to pc_raw
            flag = [False] * len(pc_raw)
            for box in value:
                box_size = np.array(box["size"])
                box_translation = np.array(box["translation"])
                box_rotation = Quaternion(np.array(box["rotation"]))
                center, orientation = trans_box_to_lidar(nusc, calib_data, ego_data, box_size=box_size,
                                                         box_center=box_translation,
                                                         box_rotation=box_rotation)

                flag = flag | get_points_in_box(pc_raw, box_size, center, orientation)

            save_name = lidar_data["filename"].split("/")[-1]

            save_pc_bin(pc_dim5, flag, save_file_dir, save_name)




