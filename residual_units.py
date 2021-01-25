from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import UpSampling2D
from keras.layers import Activation
from keras.layers import MaxPool2D
from keras.layers import Add
from keras.layers import Multiply
from keras.layers import Lambda
from keras.layers.merge import concatenate

import numpy as np
import tensorflow as tf
from keras.layers import Input
from keras.layers import Lambda, BatchNormalization, Activation
import keras.backend as K


def inception_module(layer_in, output_channels = None):
    
    ## The author replaced the residual unit with inception unit. So we tried to make the same comparison.
    ## The inception unit we used is from github:
    ## https://github.com/hakimnasaoui/Image-Scene-Classification/blob/master/Deep_Res_ception_Model.ipynb
    
    if output_channels is None:
        output_channels = layer_in.get_shape()[-1].value
    
    n = int(output_channels/128)
    f1 = int(32*n)
    f2_in = int(40*n)
    f2_out = int(64*n)
    f3_in = int(8*n)
    f3_out = int(16*n)
    f4_out = int(16*n)
    
    # 1x1 conv
    conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)
   
    # 3x3 conv
    conv3 = Conv2D(f2_in, (1,1), padding='same', activation='relu')(layer_in)
    conv3 = Conv2D(f2_out, (3,3), padding='same', activation='relu')(conv3)
    
    # 5x5 conv
    conv5 = Conv2D(f3_in, (1,1), padding='same', activation='relu')(layer_in)
    conv5 = Conv2D(f3_out, (5,5), padding='same', activation='relu')(conv5)
    
    # 3x3 max pooling
    pool = MaxPool2D((3,3), strides=(1,1), padding='same')(layer_in)
    pool = Conv2D(f4_out, (1,1), padding='same', activation='relu')(pool)
    
    # concatenate filters, assumes filters/channels last
    layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
    
    return layer_out


def attention_inception_module(input, input_channels = None, output_channels = None, encoder_depth=1,
                     p = 1, t = 2, r = 1, NAL = False, channel_attetion = False, Spatial_attention = False):
    ## attention module for inception unit
    '''
    Implementing attention module
    ## input(tf.tensor): result of last step
    ## input_channels(int): the number of filters during training, we make it
    ##                 remain the same until the last layer(output layer)
    ## output_channels(int): the number of filters used in last layer
    ## encoder_depth
    ## p(int): the number of residual units at first stage
    ## t(int): the number of residual units at trunk branch
    ## r(int): the number of residual units at soft mask branch
    '''
    ################################ Channels setting ################################
    if input_channels is None:
        input_channels = input.get_shape()[-1].value
    if output_channels is None:
        output_channels = input.get_shape()[-1].value

    #################################### Preprocess ####################################
    for _ in range(p):
        input = inception_module(input)
    pre_pros = input

    ################################### Trunk branch ###################################
    trunks = pre_pros
    for _ in range(t):
        trunks = inception_module(trunks)

    ################################# Soft mask branch #################################
    ##################### Preprocess #####################
    masks_1 = MaxPool2D(padding='same')(pre_pros)
    for _ in range(r):
        masks_1 = inception_module(masks_1)
    #################### skip connect ####################
    tmp = {}
    for j in range(encoder_depth - 1):
        out = inception_module(masks_1)
        tmp[j] = out
        masks_1 = MaxPool2D(padding='same')(masks_1)
        for _ in range(r):
            masks_1 = inception_module(masks_1)
    masks_2 = masks_1
    for j in range(encoder_depth - 1):
        for _ in range(r):
            masks_2 = inception_module(masks_2)
        masks_2 = UpSampling2D()(masks_2)
        masks_2 = Add()([masks_2, tmp[len(tmp) - j - 1]])
    #################### Post process ####################
    masks_3 = masks_2
    for _ in range(r):
        masks_3 = inception_module(masks_3)
    masks_3 = UpSampling2D()(masks_3)
    # masks_3 = BatchNormalization()(masks_3)
    # masks_3 = Activation('relu')(masks_3)
    masks_3 = Conv2D(input_channels, (1, 1))(masks_3)
    # masks_3 = BatchNormalization()(masks_3)
    # masks_3 = Activation('relu')(masks_3)
    masks_3 = Conv2D(input_channels, (1, 1))(masks_3)
    
    if channel_attetion == True:
        masks_3 = Lambda(Channel_attention)(masks_3)
    elif Spatial_attention == True:
        masks_3 = Lambda(Spatial_attention)(masks_3)
    else:
        masks_3 = Activation('sigmoid')(masks_3)

    ################################### Merge branch ###################################
    ans = Multiply()([trunks, masks_3])
    if NAL == False:
        ans = Add()([trunks, ans])
    for _ in range(p):
        ans = inception_module(ans)
    return ans


def Channel_attention(x):
    ## channel attention activation function
    return x / K.l2_normalize(x, axis = 3)

def Spatial_attention(x):
    ## spatial attention activation function
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)
    return x 


def residual_unit(input, input_channels = None, output_channels = None, kernel_size = (3,3), stride = 1):

    ## input(tf.tensor): the input from previous part
    ## input_channels(int): the number of filters during training, we make it
    ##                 remain the same until the last layer(output layer)
    ## output_channels(int): the number of filters used in last layer
    ## kernel_size(int or tuple): 2 dimensional convolution kernel
    ## stride(int): the step each convolution kernel takes

    ## if input_channel and output_channel is not announced, we force it to be the same to the original channel
    if input_channels is None and output_channels is None:
        input_channels = input.get_shape()[-1].value
        output_channels = input.get_shape()[-1].value

    if input_channels != output_channels or stride != 1:
        ## If the dimension of input does not match the dimention of the output, they can not be added togather.
        ## So we have to do a mapping here.
        x_input = Conv2D(output_channels, (1, 1), padding='same', strides=(stride, stride))(input)
    else:
        ## identical mapping:
        x_input = input

    ## multiple layers:

    ## The code of residual unit is simple. We trid to make sure it as the same to the author's as possible
    ## So we reference it from keras team's code:
    ## https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py
    
    ## first component:
    x = BatchNormalization()(input)
    x = Activation('relu')(x)
    x = Conv2D(filters = input_channels, kernel_size = (1,1), padding = "same")(x)

    ## second component:
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters = input_channels, kernel_size = kernel_size, padding = "same", strides = stride)(x)

    ## third component:
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters = output_channels, kernel_size = (1,1), padding = "same")(x)

    x = Add()([x, x_input])
    return x


def attention_module(input, input_channels = None, output_channels = None, encoder_depth=1,
                     p = 1, t = 2, r = 1, NAL = False, channel_attetion = False, Spatial_attention = False):
    
    ## We referenced some ideas of:
    ## https://github.com/qubvel/residual_attention_network
    ## and ideas from original author to construct the code below
    
    '''
    Implementing attention module
    ## input(tf.tensor): result of last step
    ## input_channels(int): the number of filters during training, we make it
    ##                 remain the same until the last layer(output layer)
    ## output_channels(int): the number of filters used in last layer
    ## encoder_depth
    ## p(int): the number of residual units at first stage
    ## t(int): the number of residual units at trunk branch
    ## r(int): the number of residual units at soft mask branch
    '''
    
    ################################ Channels setting ################################
    if input_channels is None:
        input_channels = input.get_shape()[-1].value
    if output_channels is None:
        output_channels = input.get_shape()[-1].value

    #################################### Preprocess ####################################
    for _ in range(p):
        input = residual_unit(input)
    pre_pros = input

    ################################### Trunk branch ###################################
    trunks = pre_pros
    for _ in range(t):
        trunks = residual_unit(trunks)

    ################################# Soft mask branch #################################
    ##################### Preprocess #####################
    masks_1 = MaxPool2D(padding='same')(pre_pros)
    for _ in range(r):
        masks_1 = residual_unit(masks_1)
    #################### skip connect ####################
    tmp = {}
    for j in range(encoder_depth - 1):
        out = residual_unit(masks_1)
        tmp[j] = out
        masks_1 = MaxPool2D(padding='same')(masks_1)
        for _ in range(r):
            masks_1 = residual_unit(masks_1)
    masks_2 = masks_1
    for j in range(encoder_depth - 1):
        for _ in range(r):
            masks_2 = residual_unit(masks_2)
        masks_2 = UpSampling2D()(masks_2)
        masks_2 = Add()([masks_2, tmp[len(tmp) - j - 1]])
    #################### Post process ####################
    masks_3 = masks_2
    for _ in range(r):
        masks_3 = residual_unit(masks_3)
    masks_3 = UpSampling2D()(masks_3)
    # masks_3 = BatchNormalization()(masks_3)
    # masks_3 = Activation('relu')(masks_3)
    masks_3 = Conv2D(input_channels, (1, 1))(masks_3)
    # masks_3 = BatchNormalization()(masks_3)
    # masks_3 = Activation('relu')(masks_3)
    masks_3 = Conv2D(input_channels, (1, 1))(masks_3)
    
    if channel_attetion == True:
        masks_3 = Lambda(Channel_attention)(masks_3)
    elif Spatial_attention == True:
        masks_3 = Lambda(Spatial_attention)(masks_3)
    else:
        masks_3 = Activation('sigmoid')(masks_3)

    ################################### Merge branch ###################################
    ans = Multiply()([trunks, masks_3])
    if NAL == False:
        ans = Add()([trunks, ans])
    for _ in range(p):
        ans = residual_unit(ans)
    return ans





    

