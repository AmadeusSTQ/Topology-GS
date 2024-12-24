import os
import time
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from topologylayer.nn.alpha_dionysus import AlphaLayer
# from topologylayer.nn.alpha import AlphaLayer


proj_path = '/'.join(os.path.abspath(__file__).split('/')[:-1])
is_blender = False
if not is_blender:
    data = 'db'
    # data = 'mipnerf360'
    # data = 'tandt'
    scene = 'drjohnson'
    # scene = 'treehill'
    # scene = 'truck'
    ratio = 0.08  # for db
    # ratio = 0.06  # for mipnerf360
    # ratio = 0.1  # for tandt
    image_path = proj_path + '/../data/{}/{}/images/'.format(data, scene)  # for Colmap
else:
    data = 'nerfsynthetic'
    scene = 'mic'
    ratio = 0.09  # for nerfsynthetic
    image_path = proj_path + '/../data/{}/{}/'.format(data, scene)  # for Blender

resolution_scale = 1.0
mode = 'area'
dims = [0, 1, 2]
npy_path = proj_path + '/../npy/persistence/{}/{}/'.format(data.upper(), scene.upper())
os.makedirs(npy_path, exist_ok=True)


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def clear_imgs(imgs_list):
    new_list = []
    for img in imgs_list:
        if 'depth' in img or 'norm' in img:
            continue
        new_list.append(img)
    return new_list


if __name__ == "__main__":
    # from topologylayer.nn.alpha import AlphaLayer
    # data = np.random.rand(10, 3)
    # x = torch.autograd.Variable(torch.tensor(data).type(torch.float), requires_grad=True)
    # layer = AlphaLayer(maxdim=dims[-1])
    # _ = layer(x)
    w, h = None, None
    layer = AlphaLayer(maxdim=dims[-1])
    save_npy = os.path.join(npy_path, mode + ''.join(str(ratio).split('.')) + '.npy')

    if not is_blender:
        imgs = os.listdir(image_path)
        imgs.sort()

        diag_gts = dict()

        for img_file in tqdm(imgs):
            img_name = img_file.split('.')[0]
            img_path = os.path.join(image_path, img_file)
            image = Image.open(img_path)

            if w is None and h is None:
                print(image.size)
                w, h = image.size
            if w > 1600:
                global_down = w / 1600
                scale = float(global_down) * float(resolution_scale)
                resolution = (int(w / scale), int(h / scale))
                img = PILtoTorch(image, resolution)
                img = img.unsqueeze(0)
            else:
                img = torch.from_numpy(np.array(image)) / 255.0
                img = img.permute(2, 0, 1).unsqueeze(0)

            img = F.interpolate(img, scale_factor=ratio, mode=mode).squeeze(0)

            c, h, w = img.shape
            img = img.permute(1, 2, 0).contiguous().view(-1, c)  # gt image

            # start_time = time.time()
            diag_gt = layer(img)
            # end_time = time.time()
            # print(f"Time taken: {end_time - start_time} seconds")

            diag_gt_ = dict()
            for idx, dim in enumerate(dims):
                diag_gt_['dim{}'.format(str(dim))] = diag_gt[0][idx].numpy()
            diag_gts[img_name] = (diag_gt_, diag_gt[1])

            # print('haha')
        np.save(save_npy, diag_gts)

    else:
        all_diag_gts = dict()
        for sub in ['train', 'val', 'test']:
            imgs = os.listdir(os.path.join(image_path, sub))
            imgs = clear_imgs(imgs)
            imgs.sort()

            diag_gts = dict()

            for img_file in tqdm(imgs):
                img_name = img_file.split('.')[0]
                img_path = os.path.join(image_path, sub, img_file)
                image = Image.open(img_path)

                w, h = image.size
                if w > 1600:
                    global_down = w / 1600
                    scale = float(global_down) * float(resolution_scale)
                    resolution = (int(w / scale), int(h / scale))
                    img = PILtoTorch(image, resolution)
                    img = img.unsqueeze(0)
                else:
                    img = torch.from_numpy(np.array(image)) / 255.0
                    img = img.permute(2, 0, 1).unsqueeze(0)

                img = F.interpolate(img, scale_factor=ratio, mode=mode).squeeze(0)

                c, h, w = img.shape
                img = img.permute(1, 2, 0).contiguous().view(-1, c)  # gt image

                # start_time = time.time()
                diag_gt = layer(img)
                # end_time = time.time()
                # print(f"Time taken: {end_time - start_time} seconds")

                diag_gt_ = dict()
                for idx, dim in enumerate(dims):
                    diag_gt_['dim{}'.format(str(dim))] = diag_gt[0][idx].numpy()
                diag_gts[img_name] = (diag_gt_, diag_gt[1])

                # print('haha')
            all_diag_gts[sub] = diag_gts

        np.save(save_npy, all_diag_gts)

    print('Finish!')
