import random
import cv2,os
import numpy as np
import torch
from nets.hourglass import get_hourglass
import matplotlib.pyplot as plt
from skimage.transform import resize

random.seed(None)

model_path  = "./checkpoints/140.pth"
train_img_dir = os.listdir("./data/retail/images")
images_path = "./data/retail/images/"+train_img_dir[random.randint(1,150)]

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
hmap = hmap[0].squeeze()/255.
hmap = resize(hmap.detach().numpy(), (96*4, 128*4))
plt.imshow(hmap, alpha = 0.5)

plt.show()
