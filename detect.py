import random
import cv2,os,time
import numpy as np
import torch
from nets.hourglass import get_hourglass
import matplotlib.pyplot as plt
from utils.image import transform_preds
from skimage.transform import resize
from utils.post_process import ctdet_decode, _nms, _topk

random.seed(a=None)

model_path  = "./checkpoints/140.pth"
train_img_dir = os.listdir("./data/retail/images")
#images_path = "./data/retail/images/"+train_img_dir[random.randint(1,150)]
images_path = "./data/retail/images/"+train_img_dir[120]
score_threshold = 0.9

#load a image
img = cv2.imread(images_path)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
height, width = img.shape[0], img.shape[1]

img = img.astype(np.float32) / 255.
img = img.transpose(2, 0, 1)
img = torch.tensor(img).view(-1,3,height,width)

#load model
device = torch.device('cpu')
model = get_hourglass['large_hourglass']
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.eval()

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

hmap = resize(hmap.detach().numpy(), (96*4, 128*4))
plt.imshow(hmap, alpha = 0.5)
plt.scatter(xs*4,ys*4,c='red')

plt.show()
