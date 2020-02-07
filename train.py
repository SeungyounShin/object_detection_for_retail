import os
import sys
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import numpy as np

import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

from nets.hourglass import get_hourglass
from datasets.retail import Retail

from utils.utils import _tranpose_and_gather_feature, load_model
from utils.image import transform_preds
from utils.losses import _neg_loss, _reg_loss, _theta_loss
from utils.summary import create_summary, create_logger, create_saver, DisablePrint
from utils.post_process import ctdet_decode

# Training settings
parser = argparse.ArgumentParser(description='simple_centernet45')

parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--dist', action='store_true')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--log_name', type=str, default='test')
parser.add_argument('--pretrain_name', type=str, default='pretrain')

parser.add_argument('--dataset', type=str, default='retail', choices=['coco', 'pascal','custom'])
parser.add_argument('--arch', type=str, default='large_hourglass')

parser.add_argument('--img_size', type=int, default=512)
parser.add_argument('--split_ratio', type=float, default=1.0)

parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_step', type=str, default='90,120')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_epochs', type=int, default=140)

parser.add_argument('--test_topk', type=int, default=100)

parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--val_interval', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=2)

cfg = parser.parse_args()

os.chdir(cfg.root_dir)

cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name)
cfg.pretrain_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.pretrain_name, 'checkpoint.t7')

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

cfg.lr_step = [int(s) for s in cfg.lr_step.split(',')]


def main():

  torch.manual_seed(317)
  torch.backends.cudnn.benchmark = True  # disable this if OOM at beginning of training

  num_gpus = torch.cuda.device_count()
  print("num_gpus : ",num_gpus)
  if cfg.dist:
    cfg.device = torch.device('cuda:%d' % cfg.local_rank)
    torch.cuda.set_device(cfg.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=num_gpus, rank=cfg.local_rank)
  else:
    cfg.device = torch.device('cuda')

  print('Setting up data...')
  train_dataset = Retail('./data/', 'train')
  train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=cfg.batch_size // num_gpus
                                             if cfg.dist else cfg.batch_size,
                                             shuffle=not cfg.dist,
                                             num_workers=cfg.num_workers,
                                             pin_memory=True,
                                             drop_last=True,
                                             sampler=train_sampler if cfg.dist else None)

  print('Creating model...')
  if 'hourglass' in cfg.arch:
    model = get_hourglass[cfg.arch]
  elif 'resdcn' in cfg.arch:
    model = get_pose_net(num_layers=int(cfg.arch.split('_')[-1]), num_classes=train_dataset.num_classes)
  else:
    raise NotImplementedError
  model.to(cfg.device)

  if os.path.isfile(cfg.pretrain_dir):
    model = load_model(model, cfg.pretrain_dir)

  optimizer = torch.optim.Adam(model.parameters(), cfg.lr)
  lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.lr_step, gamma=0.1)

  def train(epoch):
    print('\n Epoch: %d' % epoch)
    model.train()
    tic = time.perf_counter()
    for batch_idx, batch in enumerate(train_loader):
      for k in batch:
        if k != 'meta':
          #batch[k] = batch[k].to(device=cfg.device, non_blocking=True)
          batch[k] = batch[k]

      outputs = model(batch['image'].to(cfg.device))
      hmap, regs, w_h_, theta = zip(*outputs)

      regs  = [_tranpose_and_gather_feature(r.cpu(), batch['inds']) for r in regs]
      w_h_  = [_tranpose_and_gather_feature(r.cpu(), batch['inds']) for r in w_h_]
      theta = [_tranpose_and_gather_feature(r.cpu(), batch['inds']) for r in theta]

      hmap_loss = _neg_loss(hmap, batch['hmap'])
      reg_loss = _reg_loss(regs, batch['regs'], batch['ind_masks'])
      w_h_loss = _reg_loss(w_h_, batch['w_h_'], batch['ind_masks'])
      theta_loss = _theta_loss(theta, batch['theta'], batch['ind_masks'])
      loss = hmap_loss + 1. * reg_loss + 0.1 * w_h_loss + 1.*theta_loss

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if batch_idx % cfg.log_interval == 0:
        duration = time.perf_counter() - tic
        tic = time.perf_counter()
        print('[%d/%d-%d/%d] ' % (epoch, cfg.num_epochs, batch_idx, len(train_loader)) +
              ' hmap_loss= %.5f reg_loss= %.5f w_h_loss= %.5f theta_loss= %.5f' %
              (hmap_loss.item(), reg_loss.item(), w_h_loss.item(), theta_loss.item()) +
              ' (%d samples/sec)' % (cfg.batch_size * cfg.log_interval / duration))

        step = len(train_loader) * epoch + batch_idx

    return

  print('Starting training...')
  for epoch in range(1, cfg.num_epochs + 1):
    train(epoch)
    torch.save(model.state_dict(), './checkpoints/'+str(epoch)+".pth")
    lr_scheduler.step(epoch)  # move to here after pytorch1.1.0

  summary_writer.close()


if __name__ == '__main__':
  with DisablePrint(local_rank=cfg.local_rank):
    main()
