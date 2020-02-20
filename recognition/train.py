from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.layers import Dense, Input, Activation, Flatten
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam

from metrics import ArcFace

import time

num_epochs = 10
batch = 4

#dataloader
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


#model
input = Input(shape=(152, 152, 3))
label = Input(shape=(8,))
x = ResNet50(include_top=False,pooling="max")(input)
output = ArcFace(n_classes=8)([x, label])
model = Model([input, label], output)

model.summary()

#model compile and trainx
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

def gen():
    train_generator = train_datagen.flow_from_directory(
            './train',
            target_size=(152, 152),
            batch_size=batch,
            class_mode='categorical')

    while True:
        x,label = train_generator.next()
        yield [x, label], label

print("==train start!==")
model.fit_generator(gen(),
                    steps_per_epoch = 500//4,
                    epochs = num_epochs)
