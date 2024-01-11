import cv2
import numpy as np
import imageio.v2 as imageio

depth_path = '/home/super/datasets/KITTI-Odometry/dataset/preprocessed/00/image_2/dpt/000000.png'
# depth_path = '/home/super/nerf/slam/Co-SLAM/data/TUM/rgbd_dataset_freiburg1_desk/depth/1305031453.374112.png'


# depth = cv2.imread()
depth = np.asarray(imageio.imread(depth_path), dtype=np.int64)

print(depth.min(), depth.max(), depth.mean())
