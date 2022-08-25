import os.path
import sys
import setuptools
import nibabel as nib
from timm.models.layers import to_3tuple
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
from torchvision.utils import save_image
import tqdm
from scipy.ndimage import gaussian_filter

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model
import visdom



def main(json_path='options/swinir3d/test/test_superformer.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)
    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist


    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()
    
    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    border = opt['scale']

    
    if opt['rank'] == 0:
        option.save(opt)
    

    opt = option.dict_to_nonedict(opt)

    
    if opt['rank'] == 0:
        logger_name = 'test'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    
    for phase, dataset_opt in opt['datasets'].items():
        if phase in ['test', 'val', 'eval']:
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)
    
    
    model = define_Model(opt)
    model.load()
    model.netG.eval()

    if opt['rank'] == 0:
        logger.info(model.info_network())
    
    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_nrmse = 0.0
    avg_time = 0.0
    idx = 0
    train_size = opt["datasets"]["test"]["train_size"]

    border = 8

    for test_data in tqdm.tqdm(test_loader):
        idx+=1
        image_name_ext = os.path.basename(test_data['L_path'][0])
        img_name, ext = os.path.splitext(image_name_ext)
        img_name = img_name.split("_")[0]

        img_dir = os.path.join(opt['path']['images'], img_name)
        util.mkdir(img_dir)

        prediction = torch.zeros_like(test_data['H'])

        HR = test_data["H"] 
        H,W,D = HR.shape[2:]
        model.netG.eval()
        i = 0
        time_in = time.time() 
        for h in range(H//train_size+1):
            for w in range(W//train_size+1):
                for d in range((D//train_size)+1):
                    i+=1
                    
                    patch_L = test_data['L'][:,:,h*train_size-2*h*border:h*train_size-2*h*border+train_size,
                                                 w*train_size-2*w*border:w*train_size-2*w*border+train_size,
                                                 d*train_size-2*d*border:d*train_size-2*d*border+train_size]
                    model.feed_data({'L':patch_L}, need_H = False)
                    model.test()
                    


                    pred_crop = model.E[:,:,border:-border,border:-border, border:-border]

                    prediction[:,:, h*train_size-2*h*border+border: h*train_size-2*h*border+train_size-border,
                                    w*train_size-2*w*border+border: w*train_size-2*w*border+train_size-border,
                                    d*train_size-2*d*border+border: d*train_size-2*d*border+train_size-border] = pred_crop
        print(i)
        time_end = time.time()   
        
        E_img = util.tensor2uint(prediction)  
        H_img = util.tensor2uint(HR) 
        L_img = util.tensor2uint(test_data['L'])
 

        current_psnr = util.calculate_psnr(E_img, H_img, border=24)
        current_nrmse = util.calculate_nrmse(H_img, E_img, border=24)
        current_ssim = util.calculate_ssim_3d(H_img, E_img, border=24)
        current_time = time_end-time_in
        logger.info('{:->4d}--> {:>10s} |PSNR: {:<4.2f}dB | SSIM: {:<4.2f} | NRMSE: {:<4.2f}'.format(idx, image_name_ext, current_psnr, current_ssim, current_nrmse))
        avg_psnr += current_psnr
        avg_nrmse += current_nrmse
        avg_ssim += current_ssim
        avg_time += current_time

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx
    avg_nrmse = avg_nrmse / idx
    avg_time = current_time/idx
    logger.info('<Avg PSNR : {:<.4f}dB, Avg SSIM : {:<.4f}, Avg NRMSE: {:<.4f}, Avg time: {:<.4f}\n'.format(avg_psnr, avg_ssim, avg_nrmse, avg_time))



if __name__ == '__main__':
    main()