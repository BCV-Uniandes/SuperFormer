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
import tqdm

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model


def main(json_path='options/swinir_3d/train/train_superformer.json'):

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

    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt['scale']
    
    if opt['rank'] == 0:
        option.save(opt)

    
    opt = option.dict_to_nonedict(opt)


    if opt['rank'] == 0:
        logger_name = 'train'
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
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)


    model = define_Model(opt)
    model.init_train()
    if opt['rank'] == 0:
        logger.info(model.info_network())
        
    for epoch in range(300000): 
        for i, train_data in enumerate(train_loader):

            current_step += 1

            model.update_learning_rate(current_step)

            model.feed_data(train_data)

            model.optimize_parameters(current_step)

            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  

                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:
                avg_psnr = 0.0
                avg_ssim = 0.0
                avg_nrmse = 0.0
                time_in = time.time() 
                idx = 0
                train_size = opt["datasets"]["test"]["train_size"]
                
                for test_data in tqdm.tqdm(test_loader):
                    idx += 1
                    image_name_ext = os.path.basename(test_data['L_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)
                    img_name = img_name.split("_")[0]

                    img_dir = os.path.join(opt['path']['images'], img_name)
                    util.mkdir(img_dir)
                    
                    HR = test_data["H"] 
                    H,W,D = HR.shape[2:]
                    patches = (HR.shape[2]//opt["datasets"]["test"]["train_size"])*(HR.shape[3]//opt["datasets"]["test"]["train_size"])*(HR.shape[4]//opt["datasets"]["test"]["train_size"])
                    model.netG.eval()
                    output = torch.zeros_like(test_data['H'])
                    i=0 
                    for h in range(H//train_size):
                        for w in range(W//train_size):
                            for d in range(D//train_size):
                                patch_L = test_data['L'][:,:,h*train_size:h*train_size+train_size,
                                                                w*train_size:w*train_size+train_size,
                                                                d*train_size:d*train_size+train_size]
                                model.feed_data({'L':patch_L},need_H=False)
                                model.test()
                                output[:,:,h*train_size:h*train_size+train_size, 
                                            w*train_size:w*train_size+train_size,
                                            d*train_size:d*train_size+train_size] = model.E
                                print(i)
                                i+=1
                    E_img = util.tensor2uint(output)  
                    H_img = util.tensor2uint(HR) 

                    save_img_path = os.path.join(img_dir, '{:s}_{:d}.nii.gz'.format(img_name, current_step))
                    output_nib = nib.Nifti1Image(E_img, np.eye(4))
                    
                    current_psnr = util.calculate_psnr(E_img, H_img, border=border)
                    current_nrmse = util.calculate_nrmse(H_img, E_img, border=border)
                    current_ssim = util.calculate_ssim_3d(H_img, E_img, border=border)

                    logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))

                    avg_psnr += current_psnr
                    avg_nrmse += current_nrmse
                    avg_ssim += current_ssim

                avg_psnr = avg_psnr / idx
                avg_ssim = avg_ssim / idx
                avg_nrmse = avg_nrmse / idx
                time_end = time.time()
                time_avg = (time_end-time_in)/idx
                logger.info('<epoch:{:3d}, iter:{:8,d}, Avg PSNR : {:<.4f}dB, Avg SSIM : {:<.4f}, Avg NRMSE: {:<.4f}, Avg time: {:<.4f}\n'.format(epoch, current_step, avg_psnr, avg_ssim, avg_nrmse, time_avg))
                model.netG.train()

if __name__ == '__main__':
    main()
