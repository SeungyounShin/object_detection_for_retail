import random
import cv2,os,time
import numpy as np
import torch
from nets.hourglass import get_hourglass
import matplotlib.pyplot as plt
import matplotlib
from utils.image import transform_preds
from skimage.transform import resize
from utils.post_process import ctdet_decode, _nms, _topk
import matplotlib.patches as patches

random.seed(a=None)

model_path  = "./checkpoints/140.pth"
train_img_dir = os.listdir("./data/retail/images")
#images_path = "./data/retail/images/"+train_img_dir[random.randint(1,150)]
images_path = "./data/retail/images/"+train_img_dir[5]
score_threshold = 0.8

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

#regs
regs  = regs[0].detach()
w_h_  = w_h_[0].detach()
theta = theta[0].detach()
getVal = lambda x : (regs[0,:,int(xs[x])-1,int(ys[x])-1],
                     w_h_[0,:,int(xs[x])-1,int(ys[x])-1],
                     theta[0,:,int(xs[x])-1,int(ys[x])-1].squeeze())
bboxes = list()
for i in range(len(xs)):
    r,s,t = getVal(i)
    cntX = xs[i]*4 + r[0]
    cntY = ys[i]*4 + r[1]
    w,h = s[0],s[1]
    startX, startY = cntX-w/2, cntY-h/2
    bboxes.append([startX,startY,w,h, cntX,cntY,t])

hmap = resize(hmap.detach().numpy(), (96*4, 128*4))
plt.imshow(hmap, alpha = 0.5)
plt.scatter(xs*4,ys*4,c='red')

for box in bboxes:
    rect = patches.Rectangle((box[0],box[1]),box[2],box[3],linewidth=1,edgecolor='r',facecolor='none')
    t = matplotlib.transforms.Affine2D().rotate_around(float(box[4]), float(box[5]), float(box[6])*180/np.pi)
    rect.set_transform(t + plt.gca().transData)
    plt.gca().add_patch(rect)


plt.show()
