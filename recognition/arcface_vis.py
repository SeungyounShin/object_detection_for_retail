from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.layers import Dense, Input, Activation, Flatten
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.models import model_from_json

from metrics import ArcFace

import time
import numpy as np

import matplotlib.pyplot as plt

ckpt_path = "./checkpoints/model.h5"

#model
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

model = Model(inputs=input, outputs=x)

model.summary()

img = np.zeros((1,152,152,3))

print(model.predict(img).shape)

test_datagen = ImageDataGenerator(rescale=1./255)
def gen():
    gnr = test_datagen.flow_from_directory(
            './train',
            target_size=(152, 152),
            batch_size=1,
            class_mode='categorical')

    while True:
        x,label = gnr.next()
        yield x,label

results = list()
classes = list()
for x,label in gen():
    results.append(model.predict(x))
    classes.append(label)
