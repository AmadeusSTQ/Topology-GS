import os
import json
import numpy as np

proj_path = '/'.join(os.path.abspath(__file__).split('/')[:-1])
# data_path = 'db'
data_path = 'mipnerf360'
# data_path = 'tandt'
# data_path = 'nerfsynthetic'
# data_path = 'imw2020'
# data_path = 'bungeenerf'
method_path = 'baseline'
# method_path = 'baseline_aug'
# method_path = 'baseline_loss'
# method_path = 'baseline_aug_loss'
data_result_path = os.path.join(proj_path, 'output', data_path, method_path)


if __name__ == "__main__":
    # skip = False
    skip = True

    SSIM = []
    PSNR = []
    LPIPS = []

    SSIM_ = []
    PSNR_ = []
    LPIPS_ = []

    scenes = os.listdir(data_result_path)
    for scene in scenes:
        # # choice 1:
        # if skip and scene in ['flowers', 'treehill']:
        #     continue
        # # choice 2:  outdoor scenes
        # if skip and scene in ['room', 'counter', 'kitchen', 'bonsai']:
        #     continue
        # choice 3:  indoor scenes
        if skip and scene in ['bicycle', 'flowers', 'garden', 'stump', 'treehill']:
            continue

        json_path = os.path.join(data_result_path, scene, 'per_view.json')
        json_path_ = os.path.join(data_result_path, scene, 'results.json')

        if os.path.isfile(json_path):
            with open(json_path) as file:
                content = file.read()
                exp_results = json.loads(content)

                SSIM += list(exp_results['ours_30000']['SSIM'].values())
                PSNR += list(exp_results['ours_30000']['PSNR'].values())
                LPIPS += list(exp_results['ours_30000']['LPIPS'].values())

        if os.path.isfile(json_path_):
            with open(json_path_) as file_:
                content_ = file_.read()
                exp_results_ = json.loads(content_)

                SSIM_.append(exp_results_['ours_30000']['SSIM'])
                PSNR_.append(exp_results_['ours_30000']['PSNR'])
                LPIPS_.append(exp_results_['ours_30000']['LPIPS'])

    SSIM_score = np.sum(SSIM) / len(SSIM)
    PSNR_score = np.sum(PSNR) / len(PSNR)
    LPIPS_score = np.sum(LPIPS) / len(LPIPS)

    SSIM_score_ = np.sum(SSIM_) / len(SSIM_)
    PSNR_score_ = np.sum(PSNR_) / len(PSNR_)
    LPIPS_score_ = np.sum(LPIPS_) / len(LPIPS_)

    print('{}/{} ssim score: {}'.format(method_path, data_path, SSIM_score))
    print('{}/{} psnr score: {}'.format(method_path, data_path, PSNR_score))
    print('{}/{} lpips score: {}'.format(method_path, data_path, LPIPS_score))

    print('{}/{} ssim* score: {}'.format(method_path, data_path, SSIM_score_))
    print('{}/{} psnr* score: {}'.format(method_path, data_path, PSNR_score_))
    print('{}/{} lpips* score: {}'.format(method_path, data_path, LPIPS_score_))

