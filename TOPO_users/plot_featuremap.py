import os
import time
import glob
import torch
import numpy as np
import cv2
import torchvision.transforms.functional as tf
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from lpipsPyTorch.modules.networks import get_network
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


proj_path = '/'.join(os.path.abspath(__file__).split('/')[:-1])


class LPIPS(torch.nn.Module):
    r"""Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        net_type (str): the network type to compare the features:
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
    """
    def __init__(self, net_type: str = 'alex'):
        super(LPIPS, self).__init__()

        # pretrained network
        self.net = get_network(net_type)

    def forward(self, x: torch.Tensor, visual=True):
        feat_x = self.net(x, visual=visual)
        return feat_x


def read_image(data, scene, method, iteration):
    if method == 'gt':
        render_path = os.path.join(proj_path, '../../results', 'Scaffold-GS', 'baseline', data, scene,
                                   'test/ours_{}/gt/*.png'.format(str(iteration)))
    elif method == 'baseline':
        render_path = os.path.join(proj_path, '../../results', 'Scaffold-GS', method, data, scene,
                                   'test/ours_{}/renders/*.png'.format(str(iteration)))
    elif method == 'baseline_loss':
        render_path = os.path.join(proj_path, '../../results', 'Topology-GS', method, data, scene,
                                   'test/ours_{}/renders/*.png'.format(str(iteration)))
    else:
        raise NotImplementedError

    render_files = glob.glob(render_path)
    render_files.sort()
    render_names = [render_file.split('/')[-1].split('.')[0] for render_file in render_files]
    render_imgs = [Image.open(render_file) for render_file in render_files]
    imgs = [tf.to_tensor(render_img).unsqueeze(0)[:, :3, :, :].cuda() for render_img in render_imgs]

    return imgs, render_names


if __name__ == "__main__":

    data = 'bungeenerf'
    scene = 'rome'
    iteration = 30000
    net_type = 'vgg'
    debug = False
    visualizer = 'EigenCAM'

    methods = ['gt', 'baseline', 'baseline_loss']
    save_root = os.path.join(proj_path, 'featuremap')
    save_feature_gt = os.path.join(save_root, 'gt')
    save_feature_baseline = os.path.join(save_root, 'baseline')
    save_feature_baseline_loss = os.path.join(save_root, 'baseline_loss')
    os.makedirs(save_feature_gt, exist_ok=True)
    os.makedirs(save_feature_baseline, exist_ok=True)
    os.makedirs(save_feature_baseline_loss, exist_ok=True)

    method0 = methods[0]  # for gt
    imgs0, names0 = read_image(data, scene, method0, iteration)  # gt images data
    method1 = methods[1]  # for baseline
    imgs1, names1 = read_image(data, scene, method1, iteration)  # baseline images data
    method2 = methods[2]  # for baseline_loss
    imgs2, names2 = read_image(data, scene, method2, iteration)  # baseline_loss images data
    assert names0 == names1 == names2, "please check the number of render images"

    if debug:  # only process the first image to save time
        imgs0, imgs1, imgs2 = [imgs0[0]], [imgs1[0]], [imgs2[0]]
        names0, names1, names2 = [names0[0]], [names1[0]], [names2[0]]

    # load LPIPS net and visualizer
    lpips_net = LPIPS(net_type=net_type)
    lpips_net.eval()
    lpips_net = lpips_net.cuda()
    # target_layers = [lpips_net.net.layers[lpips_net.net.target_layers[-1]]]
    target_layers = [lpips_net.net]
    # Construct the CAM object once, and then re-use it on many images:
    if visualizer == 'EigenCAM':
        cam = EigenCAM(model=lpips_net, target_layers=target_layers, use_cuda=True)
    else:
        raise NotImplementedError

    for idx, name in enumerate(tqdm(names0)):
        # gt
        grayscale_cam0 = cam(input_tensor=imgs0[idx], eigen_smooth=True)
        grayscale_cam0 = grayscale_cam0[0, :, :]
        img_show0 = imgs0[idx].permute(0, 2, 3, 1).squeeze(dim=0).to('cpu').numpy()
        visualization0 = show_cam_on_image(img_show0, grayscale_cam0, use_rgb=True)
        # baseline
        grayscale_cam1 = cam(input_tensor=imgs1[idx], eigen_smooth=True)
        grayscale_cam1 = grayscale_cam1[0, :, :]
        img_show1 = imgs1[idx].permute(0, 2, 3, 1).squeeze(dim=0).to('cpu').numpy()
        visualization1 = show_cam_on_image(img_show1, grayscale_cam1, use_rgb=True)
        # baseline_loss
        grayscale_cam2 = cam(input_tensor=imgs2[idx], eigen_smooth=True)
        grayscale_cam2 = grayscale_cam2[0, :, :]
        img_show2 = imgs2[idx].permute(0, 2, 3, 1).squeeze(dim=0).to('cpu').numpy()
        visualization2 = show_cam_on_image(img_show2, grayscale_cam2, use_rgb=True)

        # gt
        save0 = os.path.join(save_feature_gt, '{}.png'.format(name))
        cv2.imwrite(save0, cv2.cvtColor(visualization0, cv2.COLOR_BGR2RGB))
        # baseline
        save1 = os.path.join(save_feature_baseline, '{}.png'.format(name))
        cv2.imwrite(save1, cv2.cvtColor(visualization1, cv2.COLOR_BGR2RGB))
        # baseline_loss
        save2 = os.path.join(save_feature_baseline_loss, '{}.png'.format(name))
        cv2.imwrite(save2, cv2.cvtColor(visualization2, cv2.COLOR_BGR2RGB))

    print('Finish!')
