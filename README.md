# BiTrack

Bidirectional Offline 3D Multi-Object Tracking using Camera-LiDAR Data

## Get Started

Put KITTI tracking data to the "data/kitti/tracking" directory (symbolic links are recommended).

- If you want to generate 2D-3D detection results using VoxelRCNN and SpatialEmbedding, follow the [KITTI detection](#KITTI-Detection) instructions.

- If you have generated detection files using other models, skip to [2D-3D fusion](#KITTI-2D-3D-fusion) instructions. Put 3D detection results to the "data/kitti/tracking/$split/det3d_out/$det3d_name" directory and put 2D segmentation results to the "data/kitti/tracking/$split/seg_out/$seg_name" directory. You can also use other file paths but the configuration file should be changed accordingly.

### KITTI Detection

#### KITTI 3D Object Detection

1. Build CUDA operators for 3D-IoU and PointNet++.

```shell
python setup.py develop
```

2. Download the converted model weight file from [Google Drive](https://drive.google.com/drive/folders/1OBJPBAAJPf3pEXHRlNywmAERPnHXt7tt?usp=sharing) and put it to "detection/voxel_rcnn/voxel_rcnn.pth".

3. Network inference using a specific configuration file under the "configs" directory.

```shell
python kitti_3d_detection.py $config_path $split --inference
```

4. (Optional) Average precision evaluation for cars. (1) Convert the tracking labels to the detection format. (2) Convert the detection results to the detection format. (3) Perform evaluation using the converted labels and results.

```shell
python data_processing/dataset_tracking2object.py
python data_processing/tracking2object.py $result_src $result_dst
python eval_kitti_detection.py ./data/kitti/detection/training/label_2 $result_dst
```

#### KITTI 2D Instance Segmentation

1. Download the converted model weight file from [Google Drive](https://drive.google.com/drive/folders/1OBJPBAAJPf3pEXHRlNywmAERPnHXt7tt?usp=sharing) and put it to "segmentation/spatial_embeddings/spatial_embeddings.pth".

2. Network inference using a specific configuration file under the "configs" directory.

```shell
python kitti_2d_mots.py $config_path $split
```

#### KITTI 2D-3D Fusion

1. Crop LiDAR points that are inside 3D bounding boxes.

```shell
python data_processing/crop_points.py $config_path $split
```

2. Save image shapes to json (not all KITTI images have the same shape).

```shell
python data_processing/save_img_shapes.py $config_path $split
```

3. Run the detection fusion script.

```shell
python kitti_2d_3d_det_fusion.py $config_path $split
```

### KITTI Tracking

1. Forward tracking.

```shell
python kitti_3d_tracking.py $config_path $forward_tag $split
```

2. Backward tracking.

```shell
python kitti_3d_tracking.py $config_path $backward_tag $split --backward
```

3. Trajectory fusion and refinement.

```shell
python kitti_trajectory_refinement.py $config_path $final_tag $split $foward_tag $backward_tag
```

4. Evaluation for cars.

```shell
python TrackEval/scripts/run_kitti.py --TIME_PROGRESS False --PRINT_CONFIG False --GT_FOLDER data/kitti/tracking/training --TRACKERS_FOLDER output/kitti/$split --CLASSES_TO_EVAL car --TRACKERS_TO_EVAL $tag --SPLIT_TO_EVAL $split
```
