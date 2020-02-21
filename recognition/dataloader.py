import os
import cv2
import json
import math
import numpy as np

import torch
import torch.utils.data as data
from PIL import Image, ImageFilter

import xml.etree.ElementTree as elemTree

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
    w,h = 512,384 #image size

    self.eig_val = np.array(EIGEN_VALUES, dtype=np.float32)
    self.eig_vec = np.array(EIGEN_VECTORS, dtype=np.float32)
    self.mean = np.array(MEAN, dtype=np.float32)[None, None, :]
    self.std = np.array(STD, dtype=np.float32)[None, None, :]
    self.data_rng = np.random.RandomState(123)

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
    img_id = self.ids[index]
    img_path   = self.data_dir +"/images/"     +img_id+".jpeg"
    annot_path = self.data_dir +"/annotations/"+img_id+".xml"

    tree = elemTree.parse(annot_path)
    annotations = [[float(obj.find('robndbox').find('cx').text),      #ctrX
                    float(obj.find('robndbox').find('cy').text),      #ctrY
                    float(obj.find('robndbox').find('w').text),       #W
                    float(obj.find('robndbox').find('h').text),       #H
                    float(obj.find('robndbox').find('angle').text)]   #angle
                    for obj in tree.findall('./object')]

    labels = np.array([1. for anno in annotations])
    bboxes = np.array([anno for anno in annotations], dtype=np.float32)

    if len(bboxes) == 0:
      bboxes = np.array([[0., 0., 0., 0.]], dtype=np.float32)
      labels = np.array([[0]])

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    height, width = img.shape[0], img.shape[1]
    center = np.array([width / 2., height / 2.], dtype=np.float32)  # center of image
    scale = max(height, width) * 1.0

    flipped = False
    if self.split == 'train':
      scale = scale * np.random.choice(self.rand_scales)
      w_border = get_border(self.img_size['w'], width)
      h_border = get_border(self.img_size['h'], height)
      center[0] = np.random.randint(low=w_border, high=width - w_border)
      center[1] = np.random.randint(low=h_border, high=height - h_border)

    img = img.astype(np.float32) / 255.

    #if self.split == 'train':
      #color_aug(self.data_rng, img, self.eig_val, self.eig_vec)

    #img -= self.mean
    #img /= self.std
    img = img.transpose(2, 0, 1)  # from [H, W, C] to [C, H, W]

    hmap = np.zeros((self.num_classes, self.fmap_size['h'], self.fmap_size['w']), dtype=np.float32)  # heatmap
    w_h_ = np.zeros((self.max_objs, 2), dtype=np.float32)  # width and height
    thetas = np.zeros((self.max_objs, 1), dtype=np.float32)
    regs = np.zeros((self.max_objs, 2), dtype=np.float32)  # regression
    inds = np.zeros((self.max_objs,), dtype=np.int64)
    ind_masks = np.zeros((self.max_objs,), dtype=np.uint8)
    objCnt = np.zeros((self.max_objs,2),dtype=np.float32)

    # detections = []
    for k, (rbox, label) in enumerate(zip(bboxes, labels)):
      w, h, angle = rbox[2], rbox[3], rbox[-1]
      if h > 0 and w > 0:
        obj_c = np.array([rbox[0], rbox[1]], dtype=np.float32)/float(self.down_ratio)
        objCnt[k] = obj_c
        obj_c_int = obj_c.astype(np.int32)

        radius = max(0, int(gaussian_radius((math.ceil(h), math.ceil(w)), self.gaussian_iou)))
        draw_umich_gaussian(hmap[int(label)-1], obj_c_int, radius)
        w_h_[k] = w/self.img_size['w'] , h/self.img_size['h']
        thetas[k] = angle
        regs[k] = obj_c - obj_c_int  # discretization error
        inds[k] = obj_c_int[1] * self.fmap_size['w'] + obj_c_int[0]
        ind_masks[k] = 1
        # groundtruth bounding box coordinate with class
        # detections.append([obj_c[0] - w / 2, obj_c[1] - h / 2,
        #                    obj_c[0] + w / 2, obj_c[1] + h / 2, 1, label])

    # detections = np.array(detections, dtype=np.float32) \
    #   if len(detections) > 0 else np.zeros((1, 6), dtype=np.float32)

    return {'image': img,
            'hmap': hmap, 'w_h_': w_h_, 'regs': regs, 'inds': inds, 'ind_masks': ind_masks,
            'c': center, 's': scale, 'img_id': img_id, 'theta' : thetas, 'center':objCnt}

  def __len__(self):
    return self.num_samples


if __name__ == '__main__':
  import pickle
  import sys,random
  import matplotlib
  import matplotlib.pyplot as plt
  import matplotlib.patches as patches
  # insert at 1, 0 is the script path (or '' in REPL)
  sys.path.insert(1, '/Users/seungyoun/Desktop/machine_learning/pytorch_simple_CenterNet_45-master')
  from utils.image import get_border, get_affine_transform, affine_transform, color_aug
  from utils.image import draw_umich_gaussian, gaussian_radius
  from utils.patch import patchmaker
  import scipy.misc


  dataset = Retail('/Users/seungyoun/Desktop/machine_learning/pytorch_simple_CenterNet_45-master/data/', 'train')
  dataset_make_path = "./train"

  """
  train_loader = torch.utils.data.DataLoader(dataset, batch_size=2,
                                             shuffle=False, num_workers=0,
                                             pin_memory=True, drop_last=True)

  for b in train_loader:
      print(b)
  """


  select = random.randint(1,150)
  lendata = len(dataset)
  for index in range(lendata):
      batch = dataset[index]

      img   = torch.tensor(batch['image'])
      img = img.permute(1,2,0).numpy()

      inds = batch['inds']
      w_h_ = batch['w_h_']
      regs = batch['regs']
      theta= batch['theta']
      ind_masks = batch['ind_masks']
      cntOrg = batch['center']*4.
      objs = sum(ind_masks)
      cntOrg = np.array([cntOrg[i] for i in range(objs)])

      minipatch = list()
      for i in range(objs):
          angle = theta[i]
          w = w_h_[i][0]*512.
          h = w_h_[i][1]*384.

          obj_patch = patchmaker(img,h,w,cntOrg[i][0],cntOrg[i][1],angle)
          minipatch.append(obj_patch)

      #make dataset for recognition
      for k,p in enumerate(minipatch):
          scipy.misc.imsave(dataset_make_path+"/"+str(index)+"_"+str(k)+".jpg", p)
