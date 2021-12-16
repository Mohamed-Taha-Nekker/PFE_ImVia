import cv2 as cv
import numpy as np
import nibabel
import os

from tensorflow.python.keras.callbacks import History, LearningRateScheduler
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.layers import (Input, Lambda, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, concatenate)

from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import SGD
import random

TRAIN_PATH = 'ims/train/'
TEST_PATH = 'ims/test/'

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

train_ids = sorted(os.listdir(TRAIN_PATH))

test_ids = sorted(os.listdir(TEST_PATH))

print(len(train_ids), len(test_ids))

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

print('Resizing training images and masks')
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

        for k in range(img.shape[2]):
            im = np.float32(img[:, :, k])
            ms = np.float32(msk[:, :, k])

            # Transformation en images RGB

            colored_image = cv.cvtColor(im, cv.COLOR_GRAY2RGB)
            colored_mask = cv.cvtColor(ms, cv.COLOR_GRAY2RGB)

            # Normalisation

            norm = np.zeros((800, 800))

            colored_image = cv.normalize(colored_image, norm, 0, 255, cv.NORM_MINMAX)
            colored_mask = cv.normalize(colored_mask, norm, 0, 255, cv.NORM_MINMAX)

            colored_image = resize(colored_image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            X_train[n] = colored_image

            colored_mask = cv.cvtColor(colored_mask, cv.COLOR_RGB2GRAY)
            colored_mask = np.expand_dims(
                resize(colored_mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
            y_train[n] = colored_mask

X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
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
            im = np.float32(img[:, :, k])

            # Transformation en images RGB

            colored_image = cv.cvtColor(im, cv.COLOR_GRAY2RGB)

            # Normalisation

            norm = np.zeros((800, 800))

            colored_image = cv.normalize(colored_image, norm, 0, 255, cv.NORM_MINMAX)

            colored_image = resize(colored_image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            X_test[n] = colored_image

# Jusqu'à ici on est capable de lire un fichier niftii et le transformer en RGB.
# On a également modifié la taille des images pour qu'elles soient convenables à l'entrée de U_Net

# Inputs
inputs = Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
input_dim = X_train.shape[1]
# Change integer to float and also scale pixel values
s = Lambda(lambda x: x / 255.0)(inputs)

# Contraction/Encoder path
# Block 1
c1 = Conv2D(filters=16, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(s)
c1 = Dropout(0.1)(c1)
c1 = Conv2D(filters=16, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(c1)
p1 = MaxPooling2D(pool_size=(2, 2))(c1)
# Block 2
c2 = Conv2D(filters=32, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(p1)
c2 = Dropout(0.1)(c2)
c2 = Conv2D(filters=32, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(c2)
p2 = MaxPooling2D(pool_size=(2, 2))(c2)
# Block 3
c3 = Conv2D(filters=64, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(p2)
c3 = Dropout(0.2)(c3)
c3 = Conv2D(filters=64, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(c3)
p3 = MaxPooling2D(pool_size=(2, 2))(c3)
# Block 4
c4 = Conv2D(filters=128, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(p3)
c4 = Dropout(0.2)(c4)
c4 = Conv2D(filters=128, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(c4)
p4 = MaxPooling2D(pool_size=(2, 2))(c4)
# Block 5
c5 = Conv2D(filters=256, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(p4)
c5 = Dropout(0.3)(c5)
c5 = Conv2D(filters=256, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(c5)

# Expansion/Decoder path
# Block 6
u6 = Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same')(c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(filters=128, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(u6)
c6 = Dropout(0.2)(c6)
c6 = Conv2D(filters=128, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(c6)

# Block 7
u7 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(filters=64, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(u7)
c7 = Dropout(0.2)(c7)
c7 = Conv2D(filters=64, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(c7)

# Block 8
u8 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(filters=32, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(u8)
c8 = Dropout(0.1)(c8)
c8 = Conv2D(filters=32, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(c8)

# Block 9
u9 = Conv2DTranspose(filters=16, kernel_size=(2, 2), strides=(2, 2), padding='same')(c8)
u9 = concatenate([u9, c1])
c9 = Conv2D(filters=16, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(u9)
c9 = Dropout(0.1)(c9)
c9 = Conv2D(filters=16, kernel_size=(3, 3),
            activation='relu', kernel_initializer='he_normal',
            padding='same')(c9)
# Outputs
outputs = Conv2D(filters=1, kernel_size=(1, 1),
                 activation='sigmoid')(c9)

model = Model(inputs=[inputs], outputs=[outputs])

epochs_set = 40
learning_rate = 0.1
# decay_rate = learning_rate / epochs_set
decay_rate = 0.1
momentum = 0.8

batch_size = int(input_dim / 100)

sgd = SGD(learning_rate=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])


# model.summary()

# define the learning rate change
def exp_decay(epoch):
    l_rate = learning_rate * np.exp(-decay_rate * epoch)
    return l_rate


# learning schedule callback
loss_history = History()
lr_rate = LearningRateScheduler(exp_decay)

# Callbacks
callbacks_list = [loss_history, lr_rate]

# callbacks_list = [ModelCheckpoint('ImViA.h5', verbose=1, save_best_only=True), EarlyStopping(patience=2, monitor='val_loss'), TensorBoard(log_dir='logs')]

# model_results = model.fit(X_train, y_train, validation_split=0.1, batch_size=32, epochs=epochs_set, callbacks=callbacks_list)

model_results = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs_set, callbacks=callbacks_list,
                          validation_data=(X_train, y_train))

plt.figure(figsize=[10, 6])
for key in model_results.history.keys():
    plt.plot(model_results.history[key], label=key)

plt.legend()
plt.savefig('graphe_optimisation.jpg', bbox_inches='tight')

preds_train = model.predict(X_train[:int(X_train.shape[0] * 0.9)], verbose=1)
y_true_train = y_train[:int(y_train.shape[0] * 0.9)]
preds_val = model.predict(X_train[int(X_train.shape[0] * 0.9):], verbose=1)
y_true_val = y_train[int(y_train.shape[0] * 0.9):]
preds_test = model.predict(X_test, verbose=1)

# Thresholding
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# affichage des images
i = random.randint(0, len(preds_train_t))
# i = 5
plt.figure(figsize=(8, 8))

plt.subplot(221)
imshow(X_train[i])
plt.title('Image to be Segmented')

plt.subplot(222)
imshow(y_true_train[i])
plt.title('Segmentation Ground Truth')

plt.subplot(223)
plt.imshow(np.squeeze(preds_train[i]))
plt.title('Predicted Segmentation')

plt.subplot(224)
plt.imshow(np.squeeze(preds_train_t[i]))
plt.title('Thresholded Segmentation')

plt.savefig('Résultat.jpg', bbox_inches='tight')
print("Evaluate on val data")

results = model.evaluate(X_train[int(X_train.shape[0] * 0.9):], y_train[int(y_train.shape[0] * 0.9):],
                         batch_size=batch_size)
print("Test Loss:", results[0])
print("Test Accuracy :", results[1] * 100, "%")
