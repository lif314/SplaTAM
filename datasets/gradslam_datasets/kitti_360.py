import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from natsort import natsorted

from .basedataset import GradSLAMDataset


class KITTI360Dataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        # self.input_folder = os.path.join(basedir, sequence)
        self.input_folder = "/home/super/datasets/KITTI-360"
        # 读取位姿数据
        self.pose_path = os.path.join(self.input_folder, "poses", "00.txt")
        self._calib = self._load_calibs(self.input_folder, "00")
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/sequences/00/image_2/*.png"))
        # DPT
        # KITTI & NYU models are absolute metric depth/ Standard model is inverse depth up to unkown scale and shift.
        # depth_paths = natsorted(glob.glob(f"{self.input_folder}/preprocessed/00/image_2/dpt/*.png"))
        # Lidar CompletionFormer
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/preprocessed/00/image_2/depth/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        return color_paths, depth_paths, embedding_paths
    
    @staticmethod
    def _load_calibs(data_path, seq):
        calib_file = os.path.join(data_path, "sequences", seq, "calib.txt")
        calib_file_data = {}
        with open(calib_file, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                try:
                    calib_file_data[key] = np.array([float(x) for x in value.split()], dtype=np.float32)
                except ValueError:
                    pass
        # Create 3x4 projection matrices
        P_rect_20 = np.reshape(calib_file_data['P2'], (3, 4))
        P_rect_30 = np.reshape(calib_file_data['P3'], (3, 4))

        # Compute the rectified extrinsics from cam0 to camN
        T_0 = np.eye(4, dtype=np.float32)
        T_0[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
        T_1 = np.eye(4, dtype=np.float32)
        T_1[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

        # Poses are centered around the left camera
        # T_0 = np.linalg.inv(T_0)
        T_1 = np.linalg.inv(T_1) @ T_0
        T_0 = np.eye(4, dtype=np.float32)

        K = P_rect_20[:3, :3]

        calib_file_data = {
            "K": K,
            "T_0": T_0,
            "T_1": T_1
        }

        return calib_file_data

    def load_poses(self):
        poses = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        for i in range(self.num_imgs):
            line = lines[i]
            T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
            T_w_cam0 = T_w_cam0.reshape(3, 4)
            T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
            # These poses are camera to world !!
            T_w_cam1 = T_w_cam0 @ self._calib["T_0"]
            c2w = torch.from_numpy(T_w_cam1).float()
            poses.append(c2w)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)