from keras.layers import *
from keras.models import Model



def IC(inputs, ratio=0.2):
    x = BatchNormalization(axis=3, epsilon=1.001e-5)(inputs)
    x = Dropout(ratio)(x)
    return x

def cnn_s(input_shape=(224, 224, 3), cls_nums=1000):
    inp = Input(input_shape)
    x = BatchNormalization(axis=3, epsilon=1.001e-5)(inp)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = IC(x, 0.2)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2), (2, 2))(x)

    x = IC(x, 0.2)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = IC(x, 0.2)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), (2, 2))(x)

    x = IC(x, 0.2)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = IC(x, 0.2)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = IC(x, 0.2)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2), (2, 2))(x)

    x = IC(x, 0.2)
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = IC(x, 0.2)
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = IC(x, 0.2)
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), (2, 2))(x)

    x = IC(x, 0.2)
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = IC(x, 0.2)
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = IC(x, 0.2)
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), (2, 2))(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(cls_nums,activation='softmax')(x)
    return Model(inp, x)
if __name__ == '__main__':
    m = cnn_s()
    m.summary()