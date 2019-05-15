import random
import numpy as np
import os
import cv2
import collections
import os
import torch
from itertools import repeat
from torch.utils import data
from affinity_utils import *
from rotate_utils import *
from data_utils import *
from sknw import rdp

class SpacenetDataset(data.Dataset):
    
    def __init__(self, config, seed=7, is_train=True, return_rot_gt=False):

        ## Seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        self.split =  'train' if is_train else 'val'
        self.config = config
        ### paths
        self.dir = self.config['dir']
        self.img_root = os.path.join(self.dir, 'images/')
        self.gt_root = os.path.join(self.dir, 'gt/')
        self.image_list = self.config['file']
        
        ### list of all images
        self.images = [line.rstrip('\n') for line in open(self.image_list)]
               
        ### augmentations
        self.mirror = self.config['mirror']
        
        self.crop_size = [self.config['crop_size'],self.config['crop_size']]
        self.return_rot_gt = return_rot_gt

        ### preprocess
        self.threshold = self.config['thresh']
        print 'Threshold is set to {} for {}'.format(self.threshold, self.split)
        self.angle_theta = self.config['angle_theta']
        self.mean_bgr = np.array(eval(self.config['mean']))
        self.deviation_bgr = np.array(eval(self.config['std']))
        self.normalize_type = self.config['normalize_type']
        
        ## to avoid Deadloack  between CV Threads and Pytorch Threads caused in resizing
        cv2.setNumThreads(0)

        self.files = collections.defaultdict(list)
        for f in self.images:
            self.files[self.split].append({'img': self.img_root + f + self.config['image_suffix'], 
                                            'lbl': self.gt_root + f + self.config['gt_suffix'] })
        
    def __len__(self):
        return len(self.files[self.split])
    
    def __getitem__(self, index):
                    
        image_dict = self.files[self.split][index]
        ### read each image in list
        if os.path.isfile(image_dict['img']):
            image = cv2.imread(image_dict['img']).astype(np.float)
        else:
            print 'ERROR: couldn\'t find image -> ', image_dict['img']
        
        if os.path.isfile(image_dict['lbl']):
            gt = cv2.imread(image_dict['lbl'],0).astype(np.float)
        else:
            print 'ERROR: couldn\'t find image -> ', image_dict['lbl']

        if self.split == 'train':
            image,gt = self.random_crop(image,gt,self.crop_size)
        else:
            image = cv2.resize(image,(self.crop_size[0],self.crop_size[1]),interpolation=cv2.INTER_LINEAR)
            gt = cv2.resize(gt,(self.crop_size[0],self.crop_size[1]),interpolation=cv2.INTER_LINEAR)
        
        if self.split == 'train' and index == len(self.files[self.split])-1:
            np.random.shuffle(self.files[self.split])
        
        h,w,c = image.shape
        if self.mirror == 1:
            flip = np.random.choice(2)*2-1
            image = np.ascontiguousarray(image[:, ::flip, :])
            gt = np.ascontiguousarray(gt[:,::flip])
            rotation = np.random.randint(4) * 90
            M = cv2.getRotationMatrix2D((w/2,h/2),rotation,1)
            image = cv2.warpAffine(image,M,(w,h))
            gt = cv2.warpAffine(gt,M,(w,h))
            
        gt_orig = np.copy(gt)
        gt_orig /= 255.0
        gt_orig[gt_orig<self.threshold] = 0
        gt_orig[gt_orig>=self.threshold] = 1

        keypoints = getKeypoints(gt,thresh=0.98)

        vecmap,vecmap_angles = getVectorMapsAngles((h,w),keypoints,theta=self.angle_theta,bin_size=10)
        vecmap = torch.from_numpy(self.reshapeVecMap(vecmap))
        vecmap_angles = torch.from_numpy(vecmap_angles)

        image = self.reshape(image)
        image = torch.from_numpy(np.array(image))
        
        return image,gt_orig,vecmap_angles#,junction_image
            
    def reshape(self,image):
        
        if self.normalize_type == 'Std':
            image = (image - self.mean_bgr) / (3 * self.deviation_bgr)
        elif self.normalize_type == 'MinMax':
            image = (image - self.min_bgr) / (self.max_bgr - self.min_bgr)
            image = image * 2 - 1
        elif self.normalize_type == 'Mean':
            image -= self.mean_bgr
        else:
            image = (image / 255.0)  * 2 - 1
            
        image = image.transpose(2,0,1)
        
        return image

    def reshapeVecMap(self,image):
        
        image = image.transpose(2,0,1)
        
        return image
    
    def random_crop_or_resize(self,image,gt,resize_prob):

        if random.random() < resize_prob:
            if self.crop_size[0] != image.shape[0] and self.crop_size[1] != image.shape[1]:
                image = cv2.resize(image,(self.crop_size[0],self.crop_size[1]), interpolation = cv2.INTER_LINEAR)
                gt = cv2.resize(gt,(self.crop_size[0],self.crop_size[1]), interpolation = cv2.INTER_LINEAR)
            return image, gt
        else:
            return self.random_crop(image,gt,self.crop_size)

    def random_crop(self,image,gt,size):
        
        w, h,_ = image.shape
        crop_h, crop_w = size

        start_x = np.random.randint(0,w-crop_w)
        start_y = np.random.randint(0,h-crop_h)
        
        image = image[start_x:start_x+crop_w, start_y:start_y+crop_h,:]
        gt = gt[start_x:start_x+crop_w, start_y:start_y+crop_h]
        
        return image,gt
