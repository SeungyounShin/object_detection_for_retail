import os
import cv2
import json
import math
import numpy as np

import torch
import torch.utils.data as data

#from utils.image import get_border, get_affine_transform, affine_transform, color_aug
#from utils.image import draw_umich_gaussian, gaussian_radius

NAMES = ['__background__', 'object']

MEAN = [0.40789654, 0.44719302, 0.47026115]
STD = [0.28863828, 0.27408164, 0.27809835]
EIGEN_VALUES = [0.2141788, 0.01817699, 0.00341571]
EIGEN_VECTORS = [[-0.58752847, -0.69563484, 0.41340352],
                 [-0.5832747, 0.00994535, -0.81221408],
                 [-0.56089297, 0.71832671, 0.41158938]]


class Retail(data.Dataset):
  def __init__(self, data_dir, split, split_ratio=1.0, img_size=512):
    super(Retail, self).__init__()
    self.num_classes = 1
    self.class_name = NAMES
    h,w = 512,384

    self.eig_val = np.array(EIGEN_VALUES, dtype=np.float32)
    self.eig_vec = np.array(EIGEN_VECTORS, dtype=np.float32)
    self.mean = np.array(MEAN, dtype=np.float32)[None, None, :]
    self.std = np.array(STD, dtype=np.float32)[None, None, :]

    self.split = split # 1.0
    self.data_dir = os.path.join(data_dir, "retail")
    self.img_dir = os.listdir(self.data_dir+"/images")
    self.ids = [i.split('.')[0] for i in self.img_dir]


    self.max_objs = 128
    self.padding = 127  # 31 for resnet/resdcn
    self.down_ratio = 4
    self.img_size = {'h': h, 'w': w}
    self.fmap_size = {'h': h // self.down_ratio, 'w': w // self.down_ratio}
    self.rand_scales = np.arange(0.6, 1.4, 0.1)
    self.gaussian_iou = 0.7

    self.num_samples = len(self.ids)

    print('Loaded %d %s samples' % (self.num_samples, split))

  def __getitem__(self, index):
    img_id = self.images[index]
    img_path = os.path.join(self.img_dir, self.coco.loadImgs(ids=[img_id])[0]['file_name'])

    img = cv2.imread(img_path) # 이미지를 읽어와서!!
    height, width = img.shape[0], img.shape[1]
    center = np.array([width / 2., height / 2.], dtype=np.float32)  # center of image
    scale = max(height, width) * 1.0

    img = img.astype(np.float32) / 255.
    img -= self.mean
    img /= self.std 
    img = img.transpose(2, 0, 1)  # from [H, W, C] to [C, H, W]

    return {'image': img,
            'c': center, 's': scale, 'img_id': img_id}

if __name__ == '__main__':
  from tqdm import tqdm
  import pickle

  # dataset = Retail('/Users/seungyoun/Desktop/machine_learning/pytorch_simple_CenterNet_45-master/data/', 'train')
  dataset = Retail('/Users/hyese/Desktop/kpmg/object_detection_for_retail/data/', 'train')