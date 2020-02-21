#-*- coding:utf-8 -*-

import random
import cv2,os,time
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
import matplotlib.font_manager as fm
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

#load object detection model
model_path  = "./checkpoints/1020.pth"
device = torch.device('cpu')
model = get_hourglass['large_hourglass']
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.eval()

#load recognition model
mat = databaseMat()
ids, embedding = mat.getMat()
img2vec = img2vec()
cossim = cos = nn.CosineSimilarity()

train_img_dir = os.listdir("./data/retail/images")
#images_path = "./data/retail/images/"+train_img_dir[random.randint(1,150)]

images_path = "./data/retail/images/"+train_img_dir[40]

test_images_path = "/Users/seungyoun/Downloads/test0.jpeg"
score_threshold = 0.5

start = time.time()
#load a image
img = cv2.imread(images_path)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
height, width = img.shape[0], img.shape[1]

img = img.astype(np.float32) / 255.
img = img.transpose(2, 0, 1)
img = torch.tensor(img).view(-1,3,height,width)

#infer
outputs = model(img)
hmap, regs, w_h_, theta = zip(*outputs)

#vis
fig = plt.figure(figsize=(12, 6))
img = img.view(3,height,width).permute(1,2,0).numpy()
plt.imshow(img)
hmap = torch.sigmoid(hmap[-1])
hmap = _nms(hmap)
scores, inds, clses, ys, xs = _topk(hmap, K=100)
select = scores > score_threshold
ys, xs = ys[select], xs[select]
hmap = hmap.squeeze()

#regs
regs  = regs[0].detach()
w_h_  = w_h_[0].detach()
theta = theta[0].detach()
getVal = lambda x : (regs[0,:,int(ys[x])-1,int(xs[x])-1],
                     w_h_[0,:,int(ys[x])-1,int(xs[x])-1],
                     theta[0,:,int(ys[x])-1,int(xs[x])-1].squeeze())
getVal1 = lambda x : (regs[0,:,int(ys[x]),int(xs[x])],
                     w_h_[0,:,int(ys[x]),int(xs[x])],
                     theta[0,:,int(ys[x]),int(xs[x])].squeeze())

#rbox decode
minipatch = list()
patch_start_coords = list()
bboxes = list()
for i in range(len(xs)):
    r,s,t = getVal1(i)
    cntX = xs[i]*4 + r[0]
    cntY = ys[i]*4 + r[1]
    w,h = s[0]*512,s[1]*384
    startX, startY = cntX-w/2, cntY-h/2
    bboxes.append([startX,startY,w,h, cntX,cntY,t])
    minipatch.append(patchmaker(img,h,w,cntX,cntY,t))
    patch_start_coords.append([startX,startY])

#hmap = resize(hmap.detach().numpy(), (96*4, 128*4))
im = Image.fromarray(np.uint8(hmap.detach().numpy()*255))
hmap = np.array(im.resize((96*4, 128*4), resample=0))
#plt.imshow(hmap, alpha = 0.5)
plt.scatter(xs*4,ys*4,c='red')

for box in bboxes:
    rect = patches.Rectangle((box[0],box[1]),box[2],box[3],linewidth=2,edgecolor='r',facecolor='none')
    t = matplotlib.transforms.Affine2D().rotate_around(float(box[4]), float(box[5]), float(box[6])*180/np.pi)
    rect.set_transform(t + plt.gca().transData)
    plt.gca().add_patch(rect)

end = time.time()

items = list()
for k,p in enumerate(minipatch):
    sh = p.shape
    p = torch.tensor(p.reshape(1,sh[0],sh[1],3)).float()
    embedp = img2vec.get(p.numpy()).reshape(1,-1)

    simmat = cossim(torch.tensor(embedding),torch.tensor(embedp)).detach()
    productInd = int(simmat.argmax())
    items.append(ids[productInd])
    plt.text(patch_start_coords[k][0], patch_start_coords[k][1],
            ids[productInd] , fontsize=10,fontproperties=fontprop)

end0 =time.time()

print("detection infer time   : ",end-start)
print("recognition infer time : ",end0-end)
print(items)
plt.show()
