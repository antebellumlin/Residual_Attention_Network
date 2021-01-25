from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Dense
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.models import Model
from keras.regularizers import l2

from residual_units import residual_unit
from residual_units import attention_module
from residual_units import inception_module
from residual_units import attention_inception_module

## The structure of model is the same to the author provided. However, The author does not provide the exact information of 
## hyper parameter. In addition, the author does not provided the information about parameter of CIFAR dataset. So we referenced 
## the parameter information below:
## https://github.com/fwang91/residual-attention-network
## https://github.com/qubvel/residual_attention_network


def ARL92(input, n_classes=10):

    ## input is the input dataset

    input_ = Input(shape=input.shape[1:])
    x = Conv2D(32, (5, 5), padding='same')(input_)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(3, 3), strides = 2, padding = "same")(x)

    x = residual_unit(x, input_channels=32, output_channels=128)
    x = attention_module(x, input_channels=128, output_channels=128,encoder_depth=3)

    x = residual_unit(x, input_channels=128, output_channels=256, stride=2)
    x = attention_module(x, input_channels=256, output_channels=256, encoder_depth=2)

    x = residual_unit(x, input_channels=256, output_channels=512, stride=2)
    x = attention_module(x, input_channels=512, output_channels=512, encoder_depth=1)

    x = residual_unit(x, input_channels=512, output_channels=1024)
    x = residual_unit(x, input_channels=1024, output_channels=1024)
    x = residual_unit(x, input_channels=1024, output_channels=1024)
    x = AveragePooling2D(pool_size=(4, 4), strides = 1)(x)

    x = Flatten()(x)
    output = Dense(n_classes, activation='softmax')(x)

    model = Model(input_, output)
    return model

def NAL92(input, n_classes=10):

    ## input is the input dataset

    input_ = Input(shape=input.shape[1:])
    x = Conv2D(32, (5, 5), padding='same')(input_)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(3, 3), strides = 2, padding = "same")(x)

    x = residual_unit(x, input_channels=32, output_channels=128)
    x = attention_module(x, input_channels=128, output_channels=128,encoder_depth=3, NAL = True)

    x = residual_unit(x, input_channels=128, output_channels=256, stride=2)
    x = attention_module(x, input_channels=256, output_channels=256, encoder_depth=2, NAL = True)

    x = residual_unit(x, input_channels=256, output_channels=512, stride=2)
    x = attention_module(x, input_channels=512, output_channels=512, encoder_depth=1, NAL = True)

    x = residual_unit(x, input_channels=512, output_channels=1024)
    x = residual_unit(x, input_channels=1024, output_channels=1024)
    x = residual_unit(x, input_channels=1024, output_channels=1024)
    x = AveragePooling2D(pool_size=(4, 4), strides = 1)(x)

    x = Flatten()(x)
    output = Dense(n_classes, activation='softmax')(x)

    model = Model(input_, output)
    return model

def ARL56(input, n_classes=10):

    ## input is the input dataset

    input_ = Input(shape=input.shape[1:])
    x = Conv2D(32, (5, 5), padding='same')(input_)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(3, 3), strides = 2, padding = "same")(x)

    x = residual_unit(x, input_channels=32, output_channels=128)
    x = attention_module(x, input_channels=128, output_channels=128,encoder_depth=2)

    x = residual_unit(x, input_channels=128, output_channels=256, stride=2)
    x = attention_module(x, input_channels=256, output_channels=256, encoder_depth=1)

    x = residual_unit(x, input_channels=256, output_channels=512, stride=2)
    x = attention_module(x, input_channels=512, output_channels=512, encoder_depth=1)

    x = residual_unit(x, input_channels=512, output_channels=1024)
    x = residual_unit(x, input_channels=1024, output_channels=1024)
    x = residual_unit(x, input_channels=1024, output_channels=1024)
    x = AveragePooling2D(pool_size=(4, 4), strides = 1)(x)

    x = Flatten()(x)
    output = Dense(n_classes, activation='softmax')(x)

    model = Model(input_, output)
    return model

def NAL56(input, n_classes=10):

    ## input is the input dataset

    input_ = Input(shape=input.shape[1:])
    x = Conv2D(32, (5, 5), padding='same')(input_)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(3, 3), strides = 2, padding = "same")(x)

    x = residual_unit(x, input_channels=32, output_channels=128)
    x = attention_module(x, input_channels=128, output_channels=128,encoder_depth=2, NAL = True)

    x = residual_unit(x, input_channels=128, output_channels=256, stride=2)
    x = attention_module(x, input_channels=256, output_channels=256, encoder_depth=1, NAL = True)

    x = residual_unit(x, input_channels=256, output_channels=512, stride=2)
    x = attention_module(x, input_channels=512, output_channels=512, encoder_depth=1, NAL = True)

    x = residual_unit(x, input_channels=512, output_channels=1024)
    x = residual_unit(x, input_channels=1024, output_channels=1024)
    x = residual_unit(x, input_channels=1024, output_channels=1024)
    x = AveragePooling2D(pool_size=(4, 4), strides = 1)(x)

    x = Flatten()(x)
    output = Dense(n_classes, activation='softmax')(x)

    model = Model(input_, output)
    return model

def channel_att56(input, n_classes=10):

    ## input is the input dataset

    input_ = Input(shape=input.shape[1:])
    x = Conv2D(32, (5, 5), padding='same')(input_)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(3, 3), strides = 2, padding = "same")(x)

    x = residual_unit(x, input_channels=32, output_channels=128)
    x = attention_module(x, input_channels=128, output_channels=128,encoder_depth=2,channel_attetion=True)

    x = residual_unit(x, input_channels=128, output_channels=256, stride=2)
    x = attention_module(x, input_channels=256, output_channels=256, encoder_depth=1,channel_attetion=True)

    x = residual_unit(x, input_channels=256, output_channels=512, stride=2)
    x = attention_module(x, input_channels=512, output_channels=512, encoder_depth=1,channel_attetion=True)

    x = residual_unit(x, input_channels=512, output_channels=1024)
    x = residual_unit(x, input_channels=1024, output_channels=1024)
    x = residual_unit(x, input_channels=1024, output_channels=1024)
    x = AveragePooling2D(pool_size=(4, 4), strides = 1)(x)

    x = Flatten()(x)
    output = Dense(n_classes, activation='softmax')(x)

    model = Model(input_, output)
    return model

def inception_attention(input, n_classes=10):
    
    input_ = Input(shape=input.shape[1:])
    x = Conv2D(32, (5, 5), padding='same')(input_)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(3, 3), strides = 2, padding = "same")(x)

    x = inception_module(x, output_channels=128)
    x = attention_inception_module(x, input_channels=128, output_channels=128,encoder_depth=3)

    x = inception_module(x, output_channels=256)
    x = attention_inception_module(x, input_channels=256, output_channels=256, encoder_depth=2)

    x = inception_module(x, output_channels=512)
    x = attention_inception_module(x, input_channels=512, output_channels=512, encoder_depth=1)

    x = inception_module(x, output_channels=1024)
    x = inception_module(x, output_channels=1024)
    x = inception_module(x, output_channels=1024)
    x = AveragePooling2D(pool_size=(4, 4), strides = 1)(x)

    x = Flatten()(x)
    output = Dense(n_classes, activation='softmax')(x)

    model = Model(input_, output)
    return model













