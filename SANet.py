import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

p0 = 32
def se(dInput,kernel_num):
    conv = Conv2D(kernel_num, 3, activation = 'relu', padding = 'same')(dInput)
    Fsq = GlobalAveragePooling2D()(conv)
    Fex = Dense(kernel_num) (Fsq)
    Fex = Activation('relu')(Fex)
    Fex = Dense(kernel_num) (Fex)
    Fex = Activation('sigmoid')(Fex)
    output =  multiply([conv, Fex])
    return output


def block(dInput,kernel_num):
    conv = Conv2D(kernel_num, 1, activation = 'relu', padding = 'same')(dInput)
    conv1 = Conv2D(kernel_num*2, 3, activation = 'relu', padding = 'same')(conv)
    if kernel_num>255:
        conv1 = Dropout(0.5)(conv1)
    conv2 = Conv2D(kernel_num*2, 3, activation = 'relu', padding = 'same')(conv1)
    if kernel_num>255:
        conv2 = Dropout(0.5)(conv2)
    conv2 = Conv2D(kernel_num, 1, activation = 'relu', padding = 'same')(conv2)
    conv3 = add([conv,conv2])
    conv3 = concatenate([conv3,conv], axis = 3)
    conv6 = Conv2D(kernel_num, 3, dilation_rate=(2, 2), activation = 'relu', padding="same")(conv)
    conv4 = Conv2D(kernel_num, 3, dilation_rate=(3, 3), activation = 'relu', padding="same")(conv)
    conv5 = Conv2D(kernel_num, 3, dilation_rate=(5, 5), activation = 'relu', padding="same")(conv)
    
    output = concatenate([conv3,conv6,conv4,conv5], axis = 3)
    output = Conv2D(kernel_num, 1, activation = 'relu', padding = 'same')(output)
    return output
def attention_rap_block(dInput,kernel_num):
    conv = Conv2D(kernel_num, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dInput)
    conv1 = Conv2D(kernel_num*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv)
    if kernel_num>255:
        conv1 = Dropout(0.5)(conv1)
    conv2 = Conv2D(kernel_num*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    if kernel_num>255:
        conv2 = Dropout(0.5)(conv2)
    conv2 = Conv2D(kernel_num, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv3 = add([conv,conv2])

    conv6 = Conv2D(kernel_num, 3, dilation_rate=(2, 2), activation = 'relu', padding="same", kernel_initializer = 'he_normal')(conv)
    conv4 = Conv2D(kernel_num, 3, dilation_rate=(3, 3), activation = 'relu', padding="same", kernel_initializer = 'he_normal')(conv)
    conv5 = Conv2D(kernel_num, 3, dilation_rate=(5, 5), activation = 'relu', padding="same", kernel_initializer = 'he_normal')(conv)
    
    att_input = add([conv3,conv6,conv5,conv4])
    Fc0 = GlobalAveragePooling2D()(att_input)
    Fc1 = Dense(4*kernel_num) (Fc0)

    Fc1 = Activation('relu')(Fc1)
     
    Fc2 = Dense(4*kernel_num) (Fc1)
    Fc2 = Activation('sigmoid')(Fc2)

    output = concatenate([conv3,conv6,conv5,conv4], axis = 3)
    output = multiply([output, Fc2])
    output = Conv2D(kernel_num, 1, activation = 'relu', padding = 'same')(output)
    output = add([conv,output])
    return output
def SANet(input_size, Falg_summary=False, Falg_plot_model=False, pretrained_weights = None):
    inputs = Input(input_size)
    conv1 = attention_rap_block(inputs,p0)
    se1 = se(conv1,p0)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = attention_rap_block(pool1,2*p0)
    se2 = se(conv2,2*p0)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = attention_rap_block(pool2,4*p0)
    se3 = se(conv3,4*p0)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = attention_rap_block(pool3,8*p0)
    se4 = se(conv4,8*p0)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = attention_rap_block(pool4,16*p0//2)



    up6 = Conv2D(8*p0, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv5))	
    merge6 = concatenate([se4,up6], axis = 3)
    
    
    conv6 = attention_rap_block(merge6,8*p0)

    up7 = Conv2D(4*p0, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([se3,up7], axis = 3)
    
    
    conv7 = attention_rap_block(merge7,4*p0)
    up8 = Conv2D(2*p0, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([se2,up8], axis = 3)
    
    
    conv8 = attention_rap_block(merge8,2*p0)
    
    
    
    up9 = Conv2D(p0, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([se1,up9], axis = 3)
    conv9 = attention_rap_block(merge9,p0)
   
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)
    #opt = rmsprop(lr=0.0003, decay=1e-6) 
    #opt = SGD(lr=0.001, momentum=0.9)
    #opt = SGD(lr=LearningRate, momentum=0.9, decay=LearningRate / epochs)
    #opt = Adam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    opt = Adam(lr = 1e-4)
    model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    if Falg_summary:
        model.summary()
    if Falg_plot_model:
        keras.utils.plot_model(model, to_file='logs/Unet/model.png', show_shapes=True, show_layer_names=True)
    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


