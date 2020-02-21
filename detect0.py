#-*- coding:utf-8 -*-

import random
import cv2,os,time
import numpy as np

from PIL import Image

from utils.image import transform_preds
from utils.patch import patchmaker
from utils.post_process import ctdet_decode, _nms, _topk

from recognition.model import databaseMat
from recognition.recog import img2vec
from nets.hourglass import get_hourglass

import torch.nn as nn
import torch

random.seed(a=None)
font_path = '/Library/Fonts/NanumGothic.ttf'
fontprop = fm.FontProperties(fname=font_path, size=15)

class detect():
    def __init__(self):
        self.score_threshold = 0.8

        #load object detection model
        model_path  = "./checkpoints/1020.pth"
        self.device = torch.device('cpu')
        if(torch.cuda.is_available()):
            self.device = torch.device('cuda')
        model = get_hourglass['large_hourglass']
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = model.eval()

        #load recognition model
        mat = databaseMat()
        self.ids, self.embedding = mat.getMat()
        self.img2vec = img2vec()
        self.cossim = nn.CosineSimilarity()

    def getItems(self, img_path):
        #image load
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        height, width = img.shape[0], img.shape[1]

        img = img.astype(np.float32) / 255.
        img = img.transpose(2, 0, 1)
        img = torch.tensor(img).view(-1,3,height,width).to(self.device)

        #infer
        outputs = self.model(img)
        hmap, regs, w_h_, theta = zip(*outputs)

        img = img.view(3,height,width).permute(1,2,0).numpy()
        hmap = _nms(torch.sigmoid(hmap[-1]))
        scores, inds, clses, ys, xs = _topk(hmap, K=100)
        select = scores > self.score_threshold
        ys, xs = ys[select], xs[select]
        hmap = hmap.squeeze()

        regs  = regs[0].detach()
        w_h_  = w_h_[0].detach()
        theta = theta[0].detach()
        getVal = lambda x : (regs[0,:,int(ys[x]),int(xs[x])],
                         w_h_[0,:,int(ys[x]),int(xs[x])],
                         theta[0,:,int(ys[x]),int(xs[x])].squeeze())

        #rbox decode
        minipatch = list()
        patch_start_coords = list()
        bboxes = list()
        for i in range(len(xs)):
            r,s,t = getVal(i)
            cntX = xs[i]*4 + r[0]
            cntY = ys[i]*4 + r[1]
            w,h = s[0]*512,s[1]*384
            startX, startY = cntX-w/2, cntY-h/2
            bboxes.append([startX,startY,w,h, cntX,cntY,t])
            minipatch.append(patchmaker(img,h,w,cntX,cntY,t))
            patch_start_coords.append([startX,startY])

        #get similarity
        items = list()
        for k,p in enumerate(minipatch):
            sh = p.shape
            p = torch.tensor(p.reshape(1,sh[0],sh[1],3)).float()
            embedp = self.img2vec.get(p.numpy()).reshape(1,-1)

            simmat = self.cossim(torch.tensor(self.embedding),torch.tensor(embedp)).detach()
            productInd = int(simmat.argmax())
            items.append(self.ids[productInd])

        return items


if __name__=="__main__":
    detector = detect()
    item = detector.getItems("./static/servertest/8C4_1234.jpeg")
    print(item)
