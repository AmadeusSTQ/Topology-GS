import _init_paths
import os
import numpy as np
import subprocess

# cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
# result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
# os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
# os.system('echo $CUDA_VISIBLE_DEVICES')

import cv2
import torch
import torchvision
import torch.nn.functional as F
import json
import wandb
import time
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
# from lpipsPyTorch import lpips
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim
from TOPO_users.persistence_loss import PersistLoss
from gaussian_renderer import prefilter_voxel, render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from experiments.config import cfg

# torch.set_num_threads(32)
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")


def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()

    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    
    print('Backup Finished!')


def training(dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None, ply_path=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger()
    net, opt, pipe, data = cfg.MODEL, cfg.OPTIM, cfg.PIPE, cfg.DATA
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
    scene = Scene(opt, data, net, gaussians, ply_path=ply_path, shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.ITERATIONS), desc="Training progress")
    first_iter += 1

    # define persistence loss
    ps = PersistLoss(dims=opt.PERSISTENCE.DIMS, ks=opt.PERSISTENCE.KS).cuda()

    for iteration in range(first_iter, opt.ITERATIONS + 1):
        # # network gui not available in scaffold-gs yet
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.CONVERT_SHS_PYTHON, pipe.COMPUTE_COV3D_PYTHON, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, net.SOURCE_PATH)
        #         if do_training and ((iteration < int(opt.ITERATIONS)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        bg_color = [1, 1, 1] if net.WHITE_BACKGROUND else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
        retain_grad = (iteration < opt.UPDATE_UNTIL and iteration >= 0)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad)

        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        offset_selection_mask = render_pkg["selection_mask"]
        radii = render_pkg["radii"]
        scaling = render_pkg["scaling"]
        opacity = render_pkg["neural_opacity"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        ssim_loss = (1.0 - ssim(image, gt_image))
        scaling_reg = scaling.prod(dim=1).mean()

        if iteration >= opt.PERSISTENCE.START and iteration <= opt.PERSISTENCE.END and iteration % opt.PERSISTENCE.INTERVAL == 0:

            # persistence loss
            if opt.USE_PERS_LOSS:
                image_gt = F.interpolate(gt_image.unsqueeze(0), scale_factor=opt.PERSISTENCE.RATIO,
                                         mode=opt.PERSISTENCE.MODE).squeeze(0)
                image_pred = F.interpolate(image.unsqueeze(0), scale_factor=opt.PERSISTENCE.RATIO,
                                           mode=opt.PERSISTENCE.MODE).squeeze(0)
                try:  # to avoid scipy.spatial._qhull.QhullError: QH6271 qhull topology error 20240801
                    persist_loss, _ = ps(image_pred, image_gt, viewpoint_cam.persistence)
                except:
                    print('exception: image in training dataset is {}.'.format(viewpoint_cam.image_name))
                    persist_loss = torch.tensor(0).to(torch.float).to(Ll1.device)
            else:
                persist_loss = None
        else:
            persist_loss = None

        if persist_loss is None:
            loss = (1.0 - opt.LAMBDA_DSSIM) * Ll1 + opt.LAMBDA_DSSIM * ssim_loss + 0.01 * scaling_reg
        else:
            loss = (1.0 - opt.LAMBDA_DSSIM) * Ll1 + opt.LAMBDA_DSSIM * ssim_loss + 0.01 * scaling_reg + opt.PERSISTENCE.LAMBDA * persist_loss

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.ITERATIONS:
                progress_bar.close()

            # Log and save
            training_report(opt, tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, ssim_loss, persist_loss, None, None,
                            iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), wandb, logger)
            if iteration in saving_iterations:
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            # densification
            if iteration < opt.UPDATE_UNTIL and iteration > opt.START_STAT:
                # add statis
                gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter,
                                          offset_selection_mask, voxel_visible_mask)
                # densification
                if iteration > opt.UPDATE_FROM and iteration % opt.UPDATE_INTERVAL == 0:
                    gaussians.adjust_anchor(check_interval=opt.UPDATE_INTERVAL,
                                            success_threshold=opt.SUCCESS_THRESHOLD,
                                            grad_threshold=opt.DENSIFY_GRAD_THRESHOLD,
                                            min_opacity=opt.MIN_OPACITY)

            elif iteration == opt.UPDATE_UNTIL:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                torch.cuda.empty_cache()
                    
            # Optimizer step
            if iteration < opt.ITERATIONS:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger():
    if not cfg.MODEL.MODEL_PATH:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        cfg.MODEL.MODEL_PATH = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(cfg.MODEL.MODEL_PATH))
    os.makedirs(cfg.MODEL.MODEL_PATH, exist_ok=True)
    with open(os.path.join(cfg.MODEL.MODEL_PATH, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(cfg.MODEL.MODEL_PATH)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(opt, tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, ssim_loss, persist_loss, error_maps, gt_maps,
                    elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs, wandb=None, logger=None):
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/ssim_loss', ssim_loss.item(), iteration)

        if persist_loss is not None:
            tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/persist_loss', persist_loss.item(), iteration)

        if error_maps is not None and iteration % 100 == 1:
            tb_writer.add_images(f'{dataset_name}/error_map', error_maps[0], dataformats='HWC', global_step=iteration)
            tb_writer.add_images(f'{dataset_name}/error_map_b0', error_maps[1], dataformats='HWC', global_step=iteration)
            tb_writer.add_images(f'{dataset_name}/error_map_b1', error_maps[2], dataformats='HWC', global_step=iteration)
            tb_writer.add_images(f'{dataset_name}/error_map_d0', error_maps[3], dataformats='HWC', global_step=iteration)
            tb_writer.add_images(f'{dataset_name}/error_map_d1', error_maps[4], dataformats='HWC', global_step=iteration)

        if gt_maps is not None and iteration % 100 == 1:
            tb_writer.add_images(f'{dataset_name}/gt_map', gt_maps[0], dataformats='HWC', global_step=iteration)
            tb_writer.add_images(f'{dataset_name}/gt_map_b0', gt_maps[1], dataformats='HWC', global_step=iteration)
            tb_writer.add_images(f'{dataset_name}/gt_map_b1', gt_maps[2], dataformats='HWC', global_step=iteration)
            tb_writer.add_images(f'{dataset_name}/gt_map_d0', gt_maps[3], dataformats='HWC', global_step=iteration)
            tb_writer.add_images(f'{dataset_name}/gt_map_d1', gt_maps[4], dataformats='HWC', global_step=iteration)

        if iteration <= opt.UPDATE_UNTIL:
            tb_writer.add_scalar(f'{dataset_name}/point_numbers', scene.gaussians.get_anchor.shape[0], iteration)
            grads = scene.gaussians.offset_gradient_accum / scene.gaussians.offset_denom  # [N*k, 1]
            grads[grads.isnan()] = 0.0
            grads_norm = torch.norm(grads, dim=-1)
            tb_writer.add_histogram(f'{dataset_name}/grads_norm', grads_norm, iteration)
            tb_writer.add_histogram(f'{dataset_name}/offset_denom', scene.gaussians.offset_denom, iteration)

    if wandb is not None:
        wandb.log({"train_l1_loss": Ll1, 'train_total_loss': loss, })
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                
                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 30):
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

                        if wandb:
                            render_image_list.append(image[None])
                            errormap_list.append((gt_image[None]-image[None]).abs())
                            
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            if wandb:
                                gt_image_list.append(gt_image[None])

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                
                if tb_writer:
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if wandb is not None:
                    wandb.log({f"{config['name']}_loss_viewpoint_l1_loss": l1_test, f"{config['name']}_PSNR": psnr_test})

        if tb_writer:
            # tb_writer.add_histogram(f'{dataset_name}/'+"scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', scene.gaussians.get_anchor.shape[0], iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depths")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    
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

        # depth maps
        depth_map = render_pkg["depth"].squeeze().cpu().numpy().astype(np.float32)
        depth_map_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map_color = cv2.applyColorMap(depth_map_norm.astype(np.uint8), cv2.COLORMAP_JET)
        depth_map_tensor = torch.from_numpy(depth_map_color).permute(2, 0, 1).float() / 255.0

        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth_map_tensor, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()
    
    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
        json.dump(per_view_dict, fp, indent=True)
    
    return t_list, visible_count_list


def render_sets(opt, data, net, iteration: int, pipeline, skip_train=True, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None):
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
            logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
            if wandb is not None:
                wandb.log({"train_fps": train_fps.item(), })

        if not skip_test:
            t_test_list, visible_count = render_set(net.MODEL_PATH, "test", scene.loaded_iter,
                                                    scene.getTestCameras(), gaussians, pipeline, background)
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"test_fps": test_fps, })
    
    return visible_count


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir / "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())
        
        if wandb is not None:
            wandb.log({"test_SSIMS": torch.stack(ssims).mean().item(), })
            wandb.log({"test_PSNR_final": torch.stack(psnrs).mean().item(), })
            wandb.log({"test_LPIPS": torch.stack(lpipss).mean().item(), })

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        print("")

        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)
            
            tb_writer.add_scalar(f'{dataset_name}/VISIBLE_NUMS', torch.tensor(visible_count).mean().item(), 0)
        
        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                             "PSNR": torch.tensor(psnrs).mean().item(),
                                             "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                 "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                 "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                 "VISIBLE_COUNT": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)


def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--cfg', type=str, default='experiments/scaffold/baseline.yaml', help='which experiment')
    parser.add_argument('--spath', type=str, default=None, help='which source path')
    parser.add_argument('--opath', type=str, default=None, help='which output path')
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--gpu", type=str, default='-1')

    args = parser.parse_args(sys.argv[1:])

    cfg.merge_from_file(args.cfg)

    cfg.MODEL.SOURCE_PATH = args.spath if args.spath is not None else cfg.MODEL.SOURCE_PATH
    cfg.MODEL.MODEL_PATH = args.opath if args.opath is not None else cfg.MODEL.MODEL_PATH
    model_path = cfg.MODEL.MODEL_PATH
    os.makedirs(model_path, exist_ok=True)
    # enable logging
    logger = get_logger(model_path)
    logger.info(f'args: {args}')

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')

    try:
        saveRuntimeCode(os.path.join(cfg.MODEL.MODEL_PATH, 'backup'))
    except:
        logger.info(f'save code failed~')
        
    dataset = cfg.MODEL.SOURCE_PATH.split('/')[-1]
    exp_name = cfg.MODEL.MODEL_PATH.split('/')[-2]
    
    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"{cfg.META_ARC}-{dataset}",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None
    
    logger.info("Optimizing " + cfg.MODEL.MODEL_PATH)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # training
    training(dataset, cfg.OPTIM.TEST_ITERATIONS, cfg.OPTIM.SAVE_ITERATIONS,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb, logger)
    if args.warmup:
        logger.info("\n Warmup finished! Reboot from last checkpoints")
        new_ply_path = os.path.join(cfg.MODEL.MODEL_PATH, f'point_cloud/iteration_{args.iterations}', 'point_cloud.ply')
        training(dataset, cfg.OPTIM.TEST_ITERATIONS, cfg.OPTIM.SAVE_ITERATIONS,
                 args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb=wandb, logger=logger, ply_path=new_ply_path)

    # All done
    logger.info("\nTraining complete.")

    # rendering
    logger.info(f'\nStarting Rendering~')
    visible_count = render_sets(cfg.OPTIM, cfg.DATA, cfg.MODEL, -1, cfg.PIPE, wandb=wandb, logger=logger)
    logger.info("\nRendering complete.")

    # calc metrics
    logger.info("\n Starting evaluation...")
    evaluate(cfg.MODEL.MODEL_PATH, visible_count=visible_count, wandb=wandb, logger=logger)
    logger.info("\nEvaluating complete.")
