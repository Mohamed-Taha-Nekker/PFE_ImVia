import tensorflow as tf
import numpy as np
import os
import sys
import tqdm
from tqdm import tqdm
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import sklearn.model_selection     # For using KFold
import keras.preprocessing.image   # For using image generation
import datetime                    # To measure running time
import skimage.transform           # For resizing images
import skimage.morphology          # For using image labeling

import matplotlib.pyplot as plt    # Python 2D plotting library
import random
from tqdm import tqdm
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label



TRAIN_PATH='ims/train/'
TEST_PATH='ims/test/'

train_ids = list()
for root, dirs, files in os.walk("/ims/train/", topdown=False):
    for name in dirs:
        train_ids.append(os.path.join(root, name))
test_ids = list()
for root, dirs, files in os.walk("/ims/test/", topdown=False):
    for name in dirs:
        test_ids.append(os.path.join(root, name))

IMG_HEIGHT=256
IMG_WIDTH=256
IMG_CHANNELS=3

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

print('Get train images and masks')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask

# And the same for test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

def encode(inputs):
    conv1 = layers.Conv2D(64, 3, activation = 'relu')(inputs)
    conv1 = layers.Conv2D(64, 3, activation = 'relu')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = layers.Conv2D(128, 3, activation = 'relu')(pool1)
    conv2 = layers.Conv2D(128, 3, activation = 'relu')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = layers.Conv2D(256, 3, activation = 'relu')(pool2)
    conv3 = layers.Conv2D(256, 3, activation = 'relu')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = layers.Conv2D(512, 3, activation = 'relu')(pool3)
    conv4 = layers.Conv2D(512, 3, activation = 'relu')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = layers.Conv2D(1024, 3, activation = 'relu')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation = 'relu')(conv5)
    return conv5, conv4, conv3, conv2, conv1

def decode(conv5, conv4, conv3, conv2, conv1, num_classes):
    up6 = layers.Conv2DTranspose(512, 2, strides=(2, 2))(conv5)
    crop4 = layers.Cropping2D(4)(conv4)
    concat6 = layers.Concatenate(axis=3)([crop4,up6])
    conv6 = layers.Conv2D(512, 3, activation = 'relu')(concat6)
    conv6 = layers.Conv2D(512, 3, activation = 'relu')(conv6)

    up7 = layers.Conv2DTranspose(256, 2, strides=(2, 2))(conv6)
    crop3 = layers.Cropping2D(16)(conv3)
    concat7 = layers.Concatenate(axis=3)([crop3,up7])
    conv7 = layers.Conv2D(256, 3, activation = 'relu')(concat7)
    conv7 = layers.Conv2D(256, 3, activation = 'relu')(conv7)

    up8 = layers.Conv2DTranspose(128, 2, strides=(2, 2))(conv7)
    crop2 = layers.Cropping2D(40)(conv2)
    concat8 = layers.Concatenate(axis=3)([crop2,up8])
    conv8 = layers.Conv2D(128, 3, activation = 'relu')(concat8)
    conv8 = layers.Conv2D(128, 3, activation = 'relu')(conv8)

    up9 = layers.Conv2DTranspose(64, 2, strides=(2, 2))(conv8)
    crop1 = layers.Cropping2D(88)(conv1)
    concat9 = layers.Concatenate(axis=3)([crop1,up9])
    conv9 = layers.Conv2D(64, 3, activation = 'relu')(concat9)
    conv9 = layers.Conv2D(64, 3, activation = 'relu')(conv9)
    conv10 = layers.Conv2D(num_classes, 1)(conv9)
    conv10 = layers.Softmax(axis=-1)(conv10)
    return conv10

def create_unet(input_size=(572,572,1), num_classes=2):
    inputs = layers.Input(input_size)
    conv5, conv4, conv3, conv2, conv1 = encode(inputs)
    conv10 = decode(conv5, conv4, conv3, conv2, conv1, num_classes)
    model = Model(inputs, conv10)
    model.compile(optimizer = Adam(lr=1e-4), loss='categorical_crossentropy')
    return model

model = create_unet()
model.summary()

callbacks=[tf.keras.callbacks.EarlyStopping(patience=2,monitor='val_loss'),
           tf.keras.callbacks.TensorBoard(log_dir='logs')]

model.save('Semantic segmentation model with U-Net & TF.h5')

#results=model.fit(X_train,Y_train,validation_split=0.1,batch_size=32,epochs=28,callbacks=callbacks)
