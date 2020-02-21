from efficientnet_pytorch import EfficientNet
import torch,os,math
import cv2
import numpy as np
from PIL import Image
from recognition.recog import img2vec

class feature_extract():
    def __init__(self):
        self.model = img2vec()


class databaseMat():
    def __init__(self):
        self.root = "./data/products"
        self.images_paths = os.listdir(self.root)
        self.ids = [i.split(".")[0] for i in self.images_paths]
        self.model = img2vec()
        self.frame_size = 152
    def getMat(self):
        embedding = list()
        for i in self.images_paths:
            target_path = self.root + "/" + i
            img = cv2.imread(target_path)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = np.array(img)
            roishape = img.shape
            frame = np.zeros((self.frame_size,self.frame_size,3))
            frame[self.frame_size//2-math.ceil(roishape[0]/2.):self.frame_size//2+math.floor(roishape[0]/2.),
                  self.frame_size//2-math.ceil(roishape[1]/2.):self.frame_size//2+math.floor(roishape[1]/2.),:] = img
            frame /= 255.
            frame = torch.tensor(frame.reshape(1,self.frame_size,self.frame_size,3)).float()
            vec = self.model.get(frame.numpy()).reshape(-1)
            embedding.append(vec)
        embedding = np.stack([embedding[i] for i in range(len(embedding))])
        return self.ids, embedding

if __name__ == "__main__":
    """
    model = EfficientNet.from_pretrained('efficientnet-b0')
    img = torch.randn(1,3,152,152)
    out = model.extract_features(img).view(-1)
    print(out.shape)
    """

    mat = databaseMat()
    ids, embedding = mat.getMat()
    print(embedding)
