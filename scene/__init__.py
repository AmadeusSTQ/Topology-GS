#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import numpy as np
import time
from pathlib import Path
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from TOPO_users.interpolation import interpolate_near_underlying_manifold
from TOPO_users.interpolation import interpolate_with_midpoints
from TOPO_users.interpolation import interpolate_with_tree
from TOPO_users.interpolation import interpolate_with_random

root_dir = Path(__file__).resolve().parent.parent.as_posix()


class Scene:
    gaussians : GaussianModel

    def __init__(self, opt, data, net, gaussians: GaussianModel,
                 load_iteration=None, shuffle=True, resolution_scales=[1.0], ply_path=None,
                 extra_ids=None):  # add by stq 20240702 used for HRLVI render
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = net.MODEL_PATH
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
                
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(net.SOURCE_PATH, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](opt, net.SOURCE_PATH, net.IMAGES, net.EVAL, net.LOD)
        elif os.path.exists(os.path.join(net.SOURCE_PATH, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](opt, net.SOURCE_PATH, net.WHITE_BACKGROUND, net.EVAL, ply_path=ply_path)
        else:
            assert False, "Could not recognize scene type!"

        self.gaussians.set_appearance(len(scene_info.train_cameras))
        
        if not self.loaded_iter:
            if ply_path is not None:
                with open(ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"), 'wb') as dest_file:
                    dest_file.write(src_file.read())
            else:
                with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"), 'wb') as dest_file:
                    dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # print(f'self.cameras_extent: {self.cameras_extent}')

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, net)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, net)

        if self.loaded_iter:
            self.gaussians.load_ply_sparse_gaussian(os.path.join(self.model_path,
                                                                 "point_cloud",
                                                                 "iteration_" + str(self.loaded_iter),
                                                                 "point_cloud.ply"),
                                                    extra_ids=extra_ids)
            self.gaussians.load_mlp_checkpoints(os.path.join(self.model_path,
                                                             "point_cloud",
                                                             "iteration_" + str(self.loaded_iter)))
        else:
            scene_info.point_cloud.points.tofile(os.path.join(net.MODEL_PATH, 'pc.bin'))
            print("Without interpolation, the number of PC is: ", scene_info.point_cloud.points.shape[0])
            if data.INTERPOLATION:  # use interpolation as augmentation
                data_name = net.SOURCE_PATH.split('/')[-2].upper()
                scene_name = net.SOURCE_PATH.split('/')[-1].upper()

                start_time = time.time()

                if data.INTERPOLATION_TYPE == "LPVI":
                    print("Execute LPVI interpolation!")
                    k_max = data[data_name][scene_name].K_MAX
                    k_min = data[data_name][scene_name].K_MIN
                    threshold = data[data_name][scene_name].THRESHOLD
                    bin_name = "pc_{}_{}_{}_{}.bin".format(scene_name, k_max, k_min, ''.join(str(threshold).split('.')))
                    bin_path = os.path.join(root_dir, 'bin/LPVI', data_name)
                    os.makedirs(bin_path, exist_ok=True)
                    bin_file = os.path.join(root_dir, 'bin/LPVI', data_name, bin_name)
                    if os.path.isfile(bin_file):
                        scene_points = np.fromfile(bin_file, dtype=np.float64).reshape(-1, 3)
                    else:
                        scene_points = interpolate_near_underlying_manifold(scene_info.point_cloud.points, K_max=k_max,
                                                                            K_min=k_min, threshold=threshold)
                        scene_points.tofile(bin_file)

                elif data.INTERPOLATION_TYPE == "SAGS":
                    print("Execute SAGS interpolation!")
                    k = data[data_name][scene_name].K_MAX  # abuse the K_MAX value to represent K
                    threshold = data[data_name][scene_name].THRESHOLD
                    bin_name = "pc_{}_{}_{}.bin".format(scene_name, k, ''.join(str(threshold).split('.')))
                    bin_path = os.path.join(root_dir, 'bin/SAGS', data_name)
                    os.makedirs(bin_path, exist_ok=True)
                    bin_file = os.path.join(root_dir, 'bin/SAGS', data_name, bin_name)
                    if os.path.isfile(bin_file):
                        scene_points = np.fromfile(bin_file, dtype=np.float64).reshape(-1, 3)
                    else:
                        scene_points = interpolate_with_midpoints(scene_info.point_cloud.points,
                                                                  K=k, threshold=threshold)
                        scene_points.tofile(bin_file)

                elif data.INTERPOLATION_TYPE == "DEBLUR":  # https://arxiv.org/pdf/2401.00834
                    print("Execute DEBLUR interpolation!")
                    Np = data[data_name][scene_name].K_MAX  # abuse the K_MAX value to represent Np
                    K = data[data_name][scene_name].K_MIN  # abuse the K_MIN value to represent K
                    threshold = data[data_name][scene_name].THRESHOLD
                    bin_name = "pc_{}_{}_{}_{}.bin".format(scene_name, Np, K, ''.join(str(threshold).split('.')))
                    bin_path = os.path.join(root_dir, 'bin/DEBLUR', data_name)
                    os.makedirs(bin_path, exist_ok=True)
                    bin_file = os.path.join(root_dir, 'bin/DEBLUR', data_name, bin_name)
                    if os.path.isfile(bin_file):
                        scene_points = np.fromfile(bin_file, dtype=np.float64).reshape(-1, 3)
                    else:
                        scene_points = interpolate_with_tree(scene_info.point_cloud.points, K, Np, threshold)
                        scene_points.tofile(bin_file)

                elif data.INTERPOLATION_TYPE == "RANDOM":
                    print("Execute RANDOM interpolation!")
                    Np = data[data_name][scene_name].K_MAX  # abuse the K_MAX value to represent Np
                    bin_name = "pc_{}_{}.bin".format(scene_name, Np)
                    bin_path = os.path.join(root_dir, 'bin/RANDOM', data_name)
                    os.makedirs(bin_path, exist_ok=True)
                    bin_file = os.path.join(root_dir, 'bin/RANDOM', data_name, bin_name)
                    if os.path.isfile(bin_file):
                        scene_points = np.fromfile(bin_file, dtype=np.float64).reshape(-1, 3)
                    else:
                        scene_points = interpolate_with_random(scene_info.point_cloud.points, Np)
                        scene_points.tofile(bin_file)

                else:
                    raise NotImplementedError

                end_time = time.time()
                print(f"{data.INTERPOLATION_TYPE} interpolation consumes {end_time - start_time} seconds.")
            else:
                scene_points = scene_info.point_cloud.points

            self.gaussians.create_from_pcd(scene_points, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_mlp_checkpoints(point_cloud_path)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
