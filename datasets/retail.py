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
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    annotations = self.coco.loadAnns(ids=ann_ids)
    labels = np.array([self.cat_ids[anno['category_id']] for anno in annotations])
    bboxes = np.array([anno['bbox'] for anno in annotations], dtype=np.float32)
    if len(bboxes) == 0:
      bboxes = np.array([[0., 0., 0., 0.]], dtype=np.float32)
      labels = np.array([[0]])
    bboxes[:, 2:] += bboxes[:, :2]  # xywh to xyxy

    img = cv2.imread(img_path)
    height, width = img.shape[0], img.shape[1]
    center = np.array([width / 2., height / 2.], dtype=np.float32)  # center of image
    scale = max(height, width) * 1.0

    flipped = False
    if self.split == 'train':
      scale = scale * np.random.choice(self.rand_scales)
      w_border = get_border(128, width)
      h_border = get_border(128, height)
      center[0] = np.random.randint(low=w_border, high=width - w_border)
      center[1] = np.random.randint(low=h_border, high=height - h_border)

      if np.random.random() < 0.5:
        flipped = True
        img = img[:, ::-1, :]
        center[0] = width - center[0] - 1

    trans_img = get_affine_transform(center, scale, 0, [self.img_size['w'], self.img_size['h']])
    img = cv2.warpAffine(img, trans_img, (self.img_size['w'], self.img_size['h']))

    # -----------------------------------debug---------------------------------
    # for bbox, label in zip(bboxes, labels):
    #   if flipped:
    #     bbox[[0, 2]] = width - bbox[[2, 0]] - 1
    #   bbox[:2] = affine_transform(bbox[:2], trans_img)
    #   bbox[2:] = affine_transform(bbox[2:], trans_img)
    #   bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.img_size['w'] - 1)
    #   bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.img_size['h'] - 1)
    #   cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
    #   cv2.putText(img, self.class_name[label + 1], (int(bbox[0]), int(bbox[1])),
    #               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # cv2.imshow('img', img)
    # cv2.waitKey()
    # -----------------------------------debug---------------------------------

    img = img.astype(np.float32) / 255.

    if self.split == 'train':
      color_aug(self.data_rng, img, self.eig_val, self.eig_vec)

    img -= self.mean
    img /= self.std
    img = img.transpose(2, 0, 1)  # from [H, W, C] to [C, H, W]

    trans_fmap = get_affine_transform(center, scale, 0, [self.fmap_size['w'], self.fmap_size['h']])

    hmap = np.zeros((self.num_classes, self.fmap_size['h'], self.fmap_size['w']), dtype=np.float32)  # heatmap
    w_h_ = np.zeros((self.max_objs, 2), dtype=np.float32)  # width and height
    regs = np.zeros((self.max_objs, 2), dtype=np.float32)  # regression
    inds = np.zeros((self.max_objs,), dtype=np.int64s)
    ind_masks = np.zeros((self.max_objs,), dtype=np.uint8)

    # detections = []
    for k, (bbox, label) in enumerate(zip(bboxes, labels)):
      if flipped:
        bbox[[0, 2]] = width - bbox[[2, 0]] - 1
      bbox[:2] = affine_transform(bbox[:2], trans_fmap)
      bbox[2:] = affine_transform(bbox[2:], trans_fmap)
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.fmap_size['w'] - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.fmap_size['h'] - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      if h > 0 and w > 0:
        obj_c = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        obj_c_int = obj_c.astype(np.int32)

        radius = max(0, int(gaussian_radius((math.ceil(h), math.ceil(w)), self.gaussian_iou)))
        draw_umich_gaussian(hmap[label], obj_c_int, radius)
        w_h_[k] = 1. * w, 1. * h
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
            'c': center, 's': scale, 'img_id': img_id}

  def __len__(self):
    return self.num_samples


if __name__ == '__main__':
  from tqdm import tqdm
  import pickle

  dataset = Retail('/Users/seungyoun/Desktop/machine_learning/pytorch_simple_CenterNet_45-master/data/', 'train')
