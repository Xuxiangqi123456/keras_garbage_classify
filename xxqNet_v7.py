from keras.layers import *
from keras.models import Model

from keras import callbacks


def IC(inputs, ratio=0.2):
    x = BatchNormalization(axis=3, epsilon=1.001e-5)(inputs)
    x = Dropout(ratio)(x)
    return x
def unt(x, size):
    x_in = Conv2D(size, kernel_size=(1, 1), padding='same')(x)
    x = IC(x_in, 0.2)
    x = Activation('relu')(x)

    x = IC(x, 0.2)
    x = Conv2D(size, kernel_size=(3, 3), padding='same')(x)
    y = Activation('relu')(x)

    x = concatenate([x_in, y], axis=-1)

    x = IC(x, 0.2)
    x = Conv2D(size*2, kernel_size=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    return x



def batch(m, size, pooling=0):
    if pooling:
        pool = MaxPool2D(2)
    else:
        pool = AvgPool2D(2)
    x = IC(m, 0.2)
    x = unt(x, size)
    x = IC(x, 0.2)
    x = unt(x, 2*size)

    x = pool(x)

    x = IC(x, 0.5)
    x = Conv2D(2*size, (1, 1))(x)
    x = Activation('relu')(x)
    return x
def block(x, times):
    for i in range(2, times+1, 2):
        # x = IC(x, 0.2)
        # x = unt(x, 16*i)
        x = IC(x, 0.2)
        x = unt(x, 32*i)
    x = IC(x, 0.2)
    x = Conv2D(times*16, (1, 1), padding='same')(x)
    xx_1 = batch(x, times*4, 0)
    xx_2 = batch(x, times*4, 1)
    x = concatenate([xx_1, xx_2], axis=-1)
    return x
def xxqNet_v7(input_shape=(224, 224, 3), cls_nums=1000):
    input = Input(shape=input_shape, name='input')
    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(input)
    x = Conv2D(64, 7, strides=2, use_bias=False)(x)

    x = block(x, 8)
    x = IC(x, 0.2)
    x = Conv2D(128, (1, 1), padding='same')(x)

    x = block(x, 16)
    x = IC(x, 0.2)
    x = Conv2D(256, (1, 1), padding='same')(x)

    x = block(x, 32)
    x = IC(x, 0.2)
    x = Conv2D(512, (1, 1), padding='same')(x)

    x = block(x, 16)
    x = IC(x, 0.2)
    x = Conv2D(256, (1, 1), padding='same')(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(cls_nums, activation='softmax')(x)
    return Model(input, x)

def xxq_net(input_shape=(224, 224, 3), cls_nums=1000):

    input = Input(name='batch1', shape=input_shape)

    x1 = batch(input, 16, 1)
    x2 = batch(input, 32, 1)
    x3 = batch(input, 16, 0)

    xx_1 = concatenate([x1, x2], axis=-1, name='cell_1')
    xx_1 = Conv2D(128, (1, 1), padding='same')(xx_1)
    xx_1 = unt(xx_1, 64)

    xx_1 = MaxPool2D(2)(xx_1)

    xx1 = unt(xx_1, 32)

    xx_2 = concatenate([x2, x3], axis=-1, name='cell_2')
    xx_2 = Conv2D(128, (1, 1), padding='same')(xx_2)
    xx_2 = unt(xx_2, 64)

    xx_2 = AvgPool2D(2)(xx_2)

    xx2 = unt(xx_2, 32)
    xxx_1 = concatenate([xx1, xx2], axis=-1, name='cell_3')

    x = Conv2D(256, (1, 1), padding='same')(xxx_1)

    x = unt(x, 128)

    x = MaxPool2D(2)(x)

    x = unt(x, 64)

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(cls_nums, activation='softmax')(x)
    return Model(input, x)

if __name__ == '__main__':
    model = xxq_netv1()
    model.summary()