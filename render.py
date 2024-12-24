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
import _init_paths
import sys
import os
import torch
import numpy as np
import subprocess

cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
os.system('echo $CUDA_VISIBLE_DEVICES')


import json
import time
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from scene import Scene
from gaussian_renderer import GaussianModel
from gaussian_renderer import render, prefilter_voxel
from experiments.config import cfg


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(error_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)

    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        torch.cuda.synchronize(); t_start = time.time()

        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize(); t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)

        # gts
        gt = view.original_image[0:3, :, :]

        # error maps
        errormap = (rendering - gt).abs()

        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
        json.dump(per_view_dict, fp, indent=True)

    return t_list, visible_count_list


def render_sets(opt, data, net, iteration: int, pipeline, skip_train=True, skip_test=False):
    with torch.no_grad():
        gaussians = GaussianModel(net.FEAT_DIM,
                                  net.N_OFFSETS,
                                  net.VOXEL_SIZE,
                                  net.UPDATE_DEPTH,
                                  net.UPDATE_INIT_FACTOR,
                                  net.UPDATE_HIERACHY_FACTOR,
                                  net.USE_FEAT_BANK,
                                  net.APPEARANCE_DIM,
                                  net.RATIO,
                                  net.ADD_OPACITY_DIST,
                                  net.ADD_COV_DIST,
                                  net.ADD_COLOR_DIST,
                                  )
        scene = Scene(opt, data, net, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()

        bg_color = [1, 1, 1] if net.WHITE_BACKGROUND else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(net.MODEL_PATH):
            os.makedirs(net.MODEL_PATH)

        if not skip_train:
            t_train_list, visible_count = render_set(net.MODEL_PATH, "train", scene.loaded_iter,
                                                     scene.getTrainCameras(), gaussians, pipeline, background)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            print(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')

        if not skip_test:
            t_test_list, visible_count = render_set(net.MODEL_PATH, "test", scene.loaded_iter,
                                                    scene.getTestCameras(), gaussians, pipeline, background)
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            print(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument('--cfg', type=str, default='experiments/scaffold/baseline.yaml', help='which experiment')
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    cfg.merge_from_file(args.cfg)
    model_path = cfg.MODEL.MODEL_PATH
    print("Rendering " + model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # render_sets(cfg.OPTIM, cfg.DATA, cfg.MODEL, args.iteration, cfg.PIPE, args.skip_train, args.skip_test)
    render_sets(cfg.OPTIM, cfg.DATA, cfg.MODEL, args.iteration, cfg.PIPE)
