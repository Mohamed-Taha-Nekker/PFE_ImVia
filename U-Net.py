import numpy as np
import nibabel
import os

import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from tensorflow.python.keras.callbacks import History, LearningRateScheduler
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2 as cv
from tensorflow.keras.layers import (Input, Lambda, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, concatenate)

from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import SGD
import random


class MeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)


def dice_coef(y_true, y_pred, smooth=0.00001):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)


def dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


TRAIN_PATH = 'ims/train/'
TEST_PATH = 'ims/test/'

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1

train_ids = sorted(os.listdir(TRAIN_PATH))

test_ids = sorted(os.listdir(TEST_PATH)) ## to change

print(len(train_ids), len(test_ids))

X_train = []
y_train = []
#y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

print('Resizing training images and masks')


image_slices = np.zeros((len(train_ids), 10, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):

    path_im = TRAIN_PATH + id_ + '/images/'
    path_msk = TRAIN_PATH + id_ + '/masks/'
    r = os.listdir(path_im)
    q = os.listdir(path_msk)

    for i, j in zip(r, q):

        # Afficher l'image et le masque en cours de traitement
        print(i)
        print(j)

        image_filename = os.path.join(path_im, i)
        mask_filename = os.path.join(path_msk, j)

        # Lire les images et les masques .nii

        img = nibabel.load(image_filename)
        msk = nibabel.load(mask_filename)

        # Lire ces fichiers sous forme de matrices

        img = img.get_fdata()
        msk = msk.get_fdata()

        # Parcourir chaque slice des images et masques
        #c = []
        for k in range(img.shape[2]):
            im = np.float32(img[:, :, k])
            ms = np.float32(msk[:, :, k])

            # Normalisation

            image = resize(im, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            image = 1.0 * (image - np.min(image))/(np.max(image)-np.min(image))  # normalization between 0 and 1
            X_train.append(np.expand_dims(image, 2))

            #image_slices[n][k] = colored_image

            mask = resize(ms, (IMG_HEIGHT, IMG_WIDTH), order=0, mode='constant', preserve_range=True) # nearest neighbour interpolation
            #c=colored_mask

            mask = tf.keras.utils.to_categorical(mask, 4)
            y_train.append(mask)

X_test = []
sizes_test = []

# Test images
print('Resizing test images')
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):

    path_im = TEST_PATH + id_ + '/images/'
    r = os.listdir(path_im)

    for i in r:

        # Afficher l'image en cours de traitement
        print(i)

        image_filename = os.path.join(path_im, i)

        # Lire les images .nii

        img = nibabel.load(image_filename)

        # Lire ces fichiers sous forme de matrices

        img = img.get_fdata()

        # Parcourir chaque slice des images

        for k in range(img.shape[2]):
            image = np.float32(img[:, :, k])

            # Normalisation

            image = 1.0 * (image - np.min(image)) / (np.max(image) - np.min(image))  # normalization between 0 and 1
            image = resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            X_test.append(np.expand_dims(image, 2))

# Jusqu'à ici on est capable de lire un fichier niftii et le transformer en RGB.
# On a également modifié la taille des images pour qu'elles soient convenables à l'entrée de U_Net

#
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)

# Inputs
inputs = Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
input_dim = X_train.shape[1]

# Contraction/Encoder path
# Block 1
c1 = Conv2D(filters=32, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(inputs)
#c1 = Dropout(0.1)(c1)
c1 = Conv2D(filters=32, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(c1)
p1 = MaxPooling2D(pool_size=(2, 2))(c1)
# Block 2
c2 = Conv2D(filters=32, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(p1)
#c2 = Dropout(0.1)(c2)
c2 = Conv2D(filters=32, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(c2)
p2 = MaxPooling2D(pool_size=(2, 2))(c2)
# Block 3
c3 = Conv2D(filters=32, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(p2)
#c3 = Dropout(0.2)(c3)
c3 = Conv2D(filters=32, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(c3)
p3 = MaxPooling2D(pool_size=(2, 2))(c3)
# Block 4
c4 = Conv2D(filters=32, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(p3)
#c4 = Dropout(0.2)(c4)
c4 = Conv2D(filters=32, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(c4)
p4 = MaxPooling2D(pool_size=(2, 2))(c4)
# Block 5
c5 = Conv2D(filters=16, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(p4)
#c5 = Dropout(0.3)(c5)
c5 = Conv2D(filters=16, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(c5)

# Expansion/Decoder path
# Block 6
u6 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(filters=32, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(u6)
#c6 = Dropout(0.2)(c6)
c6 = Conv2D(filters=32, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(c6)

# Block 7
u7 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(filters=32, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(u7)
#c7 = Dropout(0.2)(c7)
c7 = Conv2D(filters=32, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(c7)

# Block 8
u8 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(filters=32, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(u8)
#c8 = Dropout(0.1)(c8)
c8 = Conv2D(filters=32, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(c8)

# Block 9
u9 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(c8)
u9 = concatenate([u9, c1])
c9 = Conv2D(filters=32, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(u9)
#c9 = Dropout(0.1)(c9)
c9 = Conv2D(filters=16, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(c9)
# Outputs
outputs = Conv2D(filters=4, kernel_size=(1, 1),
                 activation='softmax')(c9)  # filters = 4 , softmax

model = Model(inputs=[inputs], outputs=[outputs])

epochs_set = 10
learning_rate = 0.001
# decay_rate = learning_rate / epochs_set
decay_rate = 0.1
momentum = 0.8

batch_size = int(input_dim / 100)

sgd = SGD(learning_rate=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=dice_loss,
              metrics=[tf.keras.metrics.CategoricalAccuracy(), MeanIoU(name='IOU', num_classes=4)])

# model.summary()

# define the learning rate change


def exp_decay(epoch):
    l_rate = learning_rate * np.exp(-decay_rate * epoch)
    return l_rate


# learning schedule callback
loss_history = History()
lr_rate = LearningRateScheduler(exp_decay)

# Callbacks
callbacks_list = [ModelCheckpoint('ImViA.h5', verbose=1, save_best_only=True), loss_history, lr_rate]

#callbacks_list = [ModelCheckpoint('ImViA.h5', verbose=1, save_best_only=True), EarlyStopping(patience=2, monitor='val_loss'), TensorBoard(log_dir='logs')]

#model_results = model.fit(X_train, y_train, validation_split=0.1, batch_size=batch_size, epochs=epochs_set, callbacks=callbacks_list)

model_results = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs_set, callbacks=callbacks_list,
                          validation_data=(X_train, y_train))

plt.figure(figsize=[10, 6])
for key in model_results.history.keys():
    plt.plot(model_results.history[key], label=key)

plt.legend()
plt.savefig('Graphe_optimisation.jpg', bbox_inches='tight')

preds_train = model.predict(X_train[:int(X_train.shape[0]* 0.9)], verbose=1)
y_true_train = y_train[:int(y_train.shape[0] * 0.9)]
preds_val = model.predict(X_train[int(X_train.shape[0] * 0.9):], verbose=1)
y_true_val = y_train[int(y_train.shape[0] * 0.9):]
preds_test = model.predict(X_test, verbose=1)

# Thresholding
#preds_train_t = (preds_train > 0.5).astype(np.uint8)
#preds_val_t = (preds_val > 0.5).astype(np.uint8)
#preds_test_t = (preds_test > 0.5).astype(np.uint8)

preds_train_t = np.argmax(preds_train, axis=3)
preds_val_t = np.argmax(preds_val, axis=3)
preds_test_t = np.argmax(preds_test, axis=3)

# Affichage des images
i = random.randint(0, len(preds_train_t))
plt.figure(figsize=(8, 8))

plt.subplot(221)
imshow(X_train[i])
plt.title('Image')

plt.subplot(222)
imshow(np.argmax(y_true_train[i], axis=2))
plt.title('Mask')

plt.subplot(223)
plt.imshow(preds_train_t[i])
plt.title('Predicted Segmentation')

plt.savefig('Résultat.jpg', bbox_inches='tight')
print("Evaluate on val data")

results = model.evaluate(X_train[int(X_train.shape[0] * 0.9):], y_train[int(y_train.shape[0] * 0.9):],
                         batch_size=batch_size)
print("Test Loss:", results[0])
print("Test Accuracy :", results[1] * 100, "%")
