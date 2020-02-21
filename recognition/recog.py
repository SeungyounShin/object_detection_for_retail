from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.layers import Dense, Input, Activation, Flatten
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.models import model_from_json

from recognition.metrics import ArcFace

import time
import numpy as np

ckpt_path = "./recognition/checkpoints/model.h5"

#model
class img2vec():
    def __init__(self):
        input  = Input(shape=(152, 152, 3))
        label  = Input(shape=(8,))
        x      = ResNet50(include_top=False,pooling="max")(input)
        output = ArcFace(n_classes=8)([x, label])
        model  = Model([input, label], output)

        model.compile(loss='categorical_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])
        # Load the model and make modifications to it

        model.load_weights(ckpt_path)

        self.model = Model(inputs=input, outputs=x)

    def get(self,img):
    
        return self.model.predict(img)
