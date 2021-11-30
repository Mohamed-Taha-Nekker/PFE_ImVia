import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model


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
