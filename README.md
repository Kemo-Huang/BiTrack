# BiTrack

## KITTI Dataset

Put KITTI tracking data to the "data/kitti/tracking" directory (symbolic links are recommended).

If you want to generate 2D-3D detection results using VoxelRCNN and SpatialEmbedding, follow the [KITTI detection instructions](#KITTI-Detection).

If you have generate detection files using other models, skip to [KITTI tracking instructions](#KITTI-Tracking). Put 3D detection results to the "data/kitti/tracking/$split/det3d_out/$det3d_name" directory and put 2D segmentation results to the "data/kitti/tracking/$split/seg_out/$seg_name" directory. You can also use other file paths but the configuration file should be changed accordingly.

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

4. (Optional) Average precision evaluation for cars. (1) Convert the tracking sequences to the detection format, where symbolic links can be used but the system authority is required. (2) Convert the detection results to the detection format. (3) Perform evaluation using the converted labels and results.

```shell
python data_processing/dataset_tracking2object.py --symlink
python data_processing/tracking2object.py $result_src $result_dst
python eval_kitti_detection.py ./data/kitti/detection/training/label_2 $result_dst
```

#### KITTI 2D Instance Segmentation

1. Download the converted model weight file from [Google Drive](https://drive.google.com/drive/folders/1OBJPBAAJPf3pEXHRlNywmAERPnHXt7tt?usp=sharing) and put it to "segmentation/spatial_embeddings/spatial_embeddings.pth".

2. Network inference using a specific configuration file under the "configs" directory.

```shell
python kitti_2d_mots.py $config_path $split
```

### KITTI Tracking

1. Crop LiDAR points that are inside 3D bounding boxes.

```shell
python data_processing/crop_points.py $config_path $split
```

2. Save image shapes to json (not all KITTI images have the same shape).

```shell
python data_processing/save_img_shapes.py $config_path $split
```

3. 