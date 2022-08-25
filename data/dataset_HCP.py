import random
import torch
import numpy as np
import torch.utils.data as data
import utils.utils_image as util
from torchvision.transforms.functional import normalize, to_tensor
from data.data_util import paired_paths_from_HCPfolder
import scipy.io as scio
import nibabel as nib
from data.transforms import paired_random_crop_3D
from random import randint
from data.transforms import paired_random_crop
class Dataset_HCP(data.Dataset):
    def __init__(self, opt):
        super(Dataset_HCP,self).__init__()
        self.opt = opt
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.gt_folder, self.lq_folder = opt['dataroot_H'], opt['dataroot_L']
        self.paths = paired_paths_from_HCPfolder([self.lq_folder, self.gt_folder], ['lq', 'gt'],self.opt["split"])
        self.split = self.opt["split"]
        self.limit_inf_1 = 35
        self.limit_sup_1 = 220
        self.limit_inf_2 = 30
        self.limit_sup_2 = 270
        self.limit_inf_3 = 0
        self.limit_sup_3 = 250


    def __getitem__(self, index):
        scale = self.opt['scale']
        gt_path = self.paths[index]['gt_path']
        HR = nib.load(gt_path)
        HR = (HR.get_fdata()/4095.0).astype("float32")
        lq_path = self.paths[index]['lq_path']
        LR = (scio.loadmat(lq_path)["out_final"]/4095.0).astype("float32")
        if self.opt['phase'] == 'train':
            gt_size = self.opt['crop_size']
            HR = HR[self.limit_inf_1:self.limit_sup_1, self.limit_inf_2:self.limit_sup_2, self.limit_inf_3:self.limit_sup_3]
            LR = LR[self.limit_inf_1:self.limit_sup_1, self.limit_inf_2:self.limit_sup_2, self.limit_inf_3:self.limit_sup_3]
            HR, LR = paired_random_crop_3D(HR, LR, gt_size, scale, gt_path)
        img_gt = torch.from_numpy(HR).unsqueeze(0)
        img_lq = torch.from_numpy(LR).unsqueeze(0)

        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        return {
            'L': img_lq,
            'H': img_gt,
            'L_path': lq_path,
            'H_path': gt_path
        }
    
    def __len__(self):
        return len(self.paths)
    

class HCP1200_2D_Dataset(data.Dataset):
    def __init__(self, opt):
        super(HCP1200_2D_Dataset, self).__init__()
        self.opt = opt
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.gt_folder, self.lq_folder = opt['dataroot_H'], opt['dataroot_L']
        self.paths = paired_paths_from_HCPfolder(
            [self.lq_folder, self.gt_folder], ['lq', 'gt'],
            self.opt["split"])
        self.split = self.opt["split"]

    def __getitem__(self, index):
        scale = self.opt['scale']

        gt_path = self.paths[index]['gt_path']
        HR = nib.load(gt_path)
        HR = HR.get_fdata()
        lq_path = self.paths[index]['lq_path']
        
        slice_z = randint(35, 220)
        HR = (HR[slice_z,:,:]/4095.0).astype("float32")
        LR = (scio.loadmat(lq_path)["out_final"][slice_z, :, :]/4095.0).astype("float32")

        if self.opt['phase'] == 'train':
            gt_size = self.opt['crop_size']
            
            img_gt, img_lq = paired_random_crop(HR, LR, gt_size, scale,
                                                gt_path)
            
        img_gt =to_tensor(img_gt)
        img_lq = to_tensor(img_lq)

        
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'L': img_lq,
            'H': img_gt,
            'L_path': lq_path,
            'H_path': gt_path
        }

    def __len__(self):
        return len(self.paths)



        



