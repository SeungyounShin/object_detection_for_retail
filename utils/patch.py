from scipy import ndimage
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def patchmaker(img,h,w,cntX,cntY,angle):
    theta = angle/np.pi*180.
    img_shape = np.shape(img)
    frame_size = 152
    pad = 50
    roi_pad = 3

    patch = img[int(cntY-h/2)-pad:int(cntY+h/2)+pad,int(cntX-w/2)-pad:int(cntX+w/2)+pad,:]
    pil_obj = Image.fromarray(np.uint8(patch*255))
    pil_rot = pil_obj.rotate(theta)
    roi = np.array(pil_rot)
    roishape = roi.shape
    frame = np.zeros((frame_size,frame_size,3))
    roi = roi[max(int(roishape[0]/2-h/2),0):int(roishape[0]/2+h/2)+roi_pad,
              max(int(roishape[1]/2-w/2),0):int(roishape[1]/2+w/2)+roi_pad,:]
    roishape = roi.shape

    frame[frame_size//2-math.ceil(roishape[0]/2.):frame_size//2+math.floor(roishape[0]/2.),
          frame_size//2-math.ceil(roishape[1]/2.):frame_size//2+math.floor(roishape[1]/2.),:] = roi
    frame_vis = Image.fromarray(np.uint8(frame))
    frame_vis.show()
    return frame/255.
