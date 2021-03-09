"""

@author: Aniruddha Mazumdar (IIT Guwahati, India)
"""

from keras.layers import Input, Lambda, merge
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D
from keras.utils import np_utils
from keras import optimizers
from keras import initializers
import random
from keras.constraints import Constraint
from keras.callbacks import Callback
import keras
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import model_from_json
import numpy.random as rng
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras import initializers
from keras.preprocessing.image import load_img, array_to_img, img_to_array
np.random.seed(123)
from scipy import signal
from scipy.misc import imread, imsave, imresize
#keras.backend.set_image_data_format('channels_first')




#Function to normalise and prepare univ layer kernel
def normalise(w):
	j = int(w.shape[0]/2)
	for i in range(w.shape[-1]):
		w[j,j,:,i]= 0
		wsum = w[:,:,:,i].sum()
		w[:,:,:,i]/=wsum
		w[j,j,:,i]=-1
	return w

w = np.random.rand(5,5,1,3)
wgt = normalise(w)
bias = np.zeros(3)

### SRM filter kernels

### For Fixed HPF 
#ww = np.random.rand(5,5,1,3)

w =  np.zeros([5,5,1,3])

"""first filter initializer"""
w[1,1,0,0]=-1/4.0
w[1,2,0,0]=2/4.0
w[1,3,0,0]=-1/4.0
w[2,1,0,0]=2/4.0
w[2,2,0,0]=-4/4.0
w[2,3,0,0]=2/4.0
w[3,1,0,0]=-1/4.0
w[3,2,0,0]=2/4.0
w[3,3,0,0]=-1/4.0

""" Second filter initilizer """
w[0,0,0,1]=-1/12.0
w[0,1,0,1]=2/12.0
w[0,2,0,1]=-2/12.0
w[0,3,0,1]=2/12.0
w[0,4,0,1]=-1/12.0

w[1,0,0,1]=2/12.0
w[1,1,0,1]=-6/12.0
w[1,2,0,1]=8/12.0
w[1,3,0,1]=-6/12.0
w[1,4,0,1]=2/12.0
        
w[2,0,0,1]=-2/12.0
w[2,1,0,1]=8/12.0
w[2,2,0,1]=-12/12.0
w[2,3,0,1]=8/12.0
w[2,4,0,1]=-2/12.0
        
w[3,0,0,1]=2/12.0
w[3,1,0,1]=-6/12.0
w[3,2,0,1]=8/12.0
w[3,3,0,1]=-6/12.0
w[3,4,0,1]=2/12.0

w[4,0,0,1]=-1/12.0
w[4,1,0,1]=2/12.0
w[4,2,0,1]=-2/12.0
w[4,3,0,1]=2/12.0
w[4,4,0,1]=-1/12.0

""" Third Filter """
w[2,1,0,2] = 1/2.0
w[2,2,0,2] = -2/2.0
w[2,3,0,2] = 1/2.0

bias = np.zeros(3)




#### Dice Loss
def dice_coef(y_true, y_pred, axis = (1,2), smooth = 1e-5):
    l = tf.reduce_sum(y_true, axis=axis)
    r = tf.reduce_sum(y_pred, axis=axis)
    intersection = tf.reduce_sum(y_true * y_pred, axis=axis)
    dice = (2. * intersection + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice)
    return dice


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)





def conv_mask_gt(z): 
    # Get ones for each class instead of a number -- we need that
    # for cross-entropy loss later on. Sometimes the groundtruth
    # masks have values other than 1 and 0. 
    class_labels_tensor = (z>=1)
    background_labels_tensor = (z==0)

    # Convert the boolean values into floats -- so that
    # computations in cross-entropy loss is correct
    bit_mask_class = np.float32(class_labels_tensor)
    bit_mask_background = np.float32(background_labels_tensor)
    combined_mask=[]
    combined_mask.append(bit_mask_background)
    combined_mask.append(bit_mask_class)
    #combined_mask = tf.concat(concat_dim=3, values=[bit_mask_background,bit_mask_class])		

    # Lets reshape our input so that it becomes suitable for 
    # tf.softmax_cross_entropy_with_logits with [batch_size, num_classes]
    #flat_labels = tf.reshape(tensor=combined_mask, shape=(-1, 2))	
    return combined_mask#flat_labels



def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def ResUnit(x, nFilter,conv_first=False):
    activation = 'relu'
    batch_normalization = True
    strides = 1
    if conv_first:
        activation = None
        batch_normalization = False
    

    # bottleneck residual unit
    y = resnet_layer(inputs=x,
                     num_filters=nFilter,
                     kernel_size=1,
                     strides=strides,
                     activation=activation,
                     batch_normalization=batch_normalization,
                     conv_first=conv_first)
    y = resnet_layer(inputs=y,
                     num_filters=nFilter,
                     conv_first=False)
    y = resnet_layer(inputs=y,
                     num_filters=nFilter,
                     kernel_size=1,
                     conv_first=False)
   
    x = keras.layers.add([x, y])
    return x

def noiseResidue(input_image, filter_w):
    I = np.float32(input_image[:,:,0])
    
    size = input_image.shape
    
    w1 = filter_w[:,:,0,0]
    w2 = filter_w[:,:,0,1]
    w3 = filter_w[:,:,0,2]
    
    C1 = signal.convolve2d(I, w1, boundary='symm', mode='same')
    C2 = signal.convolve2d(I, w2, boundary='symm', mode='same')
    C3 = signal.convolve2d(I, w3, boundary='symm', mode='same')
    C = np.zeros(size)
    C[:,:,0] = C1
    C[:,:,1] = C2
    C[:,:,2] = C3
    
    #C = block_reduce(C, (4,4,1), func=np.max)    #  downsamples the residual matrix
    return C








nFilter = 32
input_shape=(256, 256, 3)

## RGB Stream of the Encoder

inputs_R = Input(shape=input_shape)
layer_11 = resnet_layer(inputs=inputs_R,
                     num_filters=nFilter,
                     conv_first=True)
layer_11 = ResUnit(layer_11, nFilter)
layer_11 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(layer_11)

layer_12 = resnet_layer(inputs=layer_11,
                     num_filters=2*nFilter,
                     conv_first=False)
layer_12 = ResUnit(layer_12, 2*nFilter)
layer_12 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(layer_12)

layer_13 = resnet_layer(inputs=layer_12,
                     num_filters=4*nFilter,
                     conv_first=False)
layer_13 = ResUnit(layer_13, 4*nFilter)
layer_13 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(layer_13)

layer_14 = resnet_layer(inputs=layer_13,
                     num_filters=8*nFilter,
                     conv_first=False)
layer_14 = ResUnit(layer_14, 8*nFilter)
layer_14 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(layer_14)



## Noise Stream of the Encoder

input_shape_N = (256, 256, 1)
inputs_N = Input(shape=input_shape_N)
layer_21 = Conv2D(3, (5, 5), input_shape=input_shape, padding="same", name='constrain')(inputs_N)
layer_21 = resnet_layer(inputs=layer_21,
                     num_filters=nFilter,
                     conv_first=True)
layer_21 = ResUnit(layer_21, nFilter)
layer_21 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(layer_21)

layer_22 = resnet_layer(inputs=layer_21,
                     num_filters=2*nFilter,
                     conv_first=False)
layer_22 = ResUnit(layer_22, 2*nFilter)
layer_22 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(layer_22)

layer_23 = resnet_layer(inputs=layer_22,
                     num_filters=4*nFilter,
                     conv_first=False)
layer_23 = ResUnit(layer_23, 4*nFilter)
layer_23 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(layer_23)

layer_24 = resnet_layer(inputs=layer_23,
                     num_filters=8*nFilter,
                     conv_first=False)
layer_24 = ResUnit(layer_24, 8*nFilter)
layer_24 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(layer_24)



concatenated = keras.layers.concatenate([layer_14, layer_24])

## Fused Decoder
fused_d_layer1 = UpSampling2D(size=(2, 2))(concatenated)
fused_d_layer1 = resnet_layer(inputs=fused_d_layer1,
                     num_filters=2*nFilter,
                     conv_first=False)

fused_d_layer2 = UpSampling2D(size=(2, 2))(fused_d_layer1)
fused_d_layer2 = resnet_layer(inputs=fused_d_layer2,
                     num_filters=2*nFilter,
                     conv_first=False)


fused_d_layer3 = UpSampling2D(size=(2, 2))(fused_d_layer2)
fused_d_layer3 = resnet_layer(inputs=fused_d_layer3,
                     num_filters=1*nFilter,
                     conv_first=False)

fused_d_layer4 = UpSampling2D(size=(2, 2))(fused_d_layer3)
fused_d_layer4 = resnet_layer(inputs=fused_d_layer4,
                     num_filters=nFilter,
                     conv_first=False)


## Noise Decoder
noise_d_layer1 = UpSampling2D(size=(2, 2))(layer_24)
noise_d_layer1 = resnet_layer(inputs=noise_d_layer1,
                     num_filters=2*nFilter,
                     conv_first=False)

noise_d_layer2 = UpSampling2D(size=(2, 2))(noise_d_layer1)
noise_d_layer2 = resnet_layer(inputs=noise_d_layer2,
                     num_filters=2*nFilter,
                     conv_first=False)


noise_d_layer3 = UpSampling2D(size=(2, 2))(noise_d_layer2)
noise_d_layer3 = resnet_layer(inputs=noise_d_layer3,
                     num_filters=1*nFilter,
                     conv_first=False)

noise_d_layer4 = UpSampling2D(size=(2, 2))(noise_d_layer3)
noise_d_layer4 = resnet_layer(inputs=noise_d_layer4,
                     num_filters=nFilter,
                     conv_first=False)


## RGB Decoder
rgb_d_layer1 = UpSampling2D(size=(2, 2))(layer_14)
rgb_d_layer1 = resnet_layer(inputs=rgb_d_layer1,
                     num_filters=2*nFilter,
                     conv_first=False)

rgb_d_layer2 = UpSampling2D(size=(2, 2))(rgb_d_layer1)
rgb_d_layer2 = resnet_layer(inputs=rgb_d_layer2,
                     num_filters=2*nFilter,
                     conv_first=False)


rgb_d_layer3 = UpSampling2D(size=(2, 2))(rgb_d_layer2)
rgb_d_layer3 = resnet_layer(inputs=rgb_d_layer3,
                     num_filters=1*nFilter,
                     conv_first=False)

rgb_d_layer4 = UpSampling2D(size=(2, 2))(rgb_d_layer3)
rgb_d_layer4 = resnet_layer(inputs=rgb_d_layer4,
                     num_filters=nFilter,
                     conv_first=False)

decoder_concatenated = keras.layers.concatenate([noise_d_layer4, rgb_d_layer4])

fused_decoder = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(decoder_concatenated)

# Instantiate model.
model = Model(inputs=[inputs_N, inputs_R], outputs=fused_decoder)

optimizer = keras.optimizers.Adam(lr=0.0005)

model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

print(model.summary())
    
model.get_layer('constrain').set_weights([wgt, bias])

layer_output = model.get_layer('constrain').get_weights()



###################################################################################################
###### Load the pre-trained model for resuming/finetuning the training
weight_path = './models/loss/model_weights.hd5'
#model.load_weights(weight_path)


###################################################################################################3

batch_size = 4
test_batch_size = 4
num_iters = 20000


img_path1 = './Data/Bernard/Computer_Vision/spliced_NIST/rgb_imgs/'
mask_path1 = './Data/Bernard/Computer_Vision/spliced_NIST/masks/'

img_path2 = './Data/Bernard/Computer_Vision/spliced_copymove_NIST/rgb_imgs/'
mask_path2 = './Data/Bernard/Computer_Vision/spliced_copymove_NIST/masks/'


imgDir1 = os.listdir(img_path1)

imgDirTrain1 = imgDir1[:len(imgDir1)-1000]
imgDirTest1 = imgDir1[len(imgDir1)-1000:]

imgDir2 = os.listdir(img_path2)

imgDirTrain2 = imgDir2[:len(imgDir2)-1000]
imgDirTest2 = imgDir2[len(imgDir2)-1000:]



batch_x = np.zeros((2*batch_size, 256, 256, 3))
batch_y = np.zeros((2*batch_size,256,256, 1))
batch_x1 = np.zeros((2*batch_size,256,256, 1))

batch_x_val = np.zeros((2*test_batch_size, 256, 256, 3))
batch_y_val = np.zeros((2*test_batch_size,256,256, 1))
batch_x1_val = np.zeros((2*test_batch_size,256,256, 1))


###########  For Splice Folder  ##########################

for i in range(test_batch_size):
    name = imgDirTest1[i]
    imgFilename = img_path1 + name
    img = imread(imgFilename)
    maskFilename = mask_path1 + name[:-7] + 'mask.png'
    mask = imread(maskFilename)
    
    img = imresize(img, [256,256,3])
    mask = imresize(mask, [256, 256])
    
    mask_size = mask.shape
    
    
    batch_x_val[i,:,:,:] = img[:,:,0:3]
    batch_x1_val[i,:,:,0] = img[:,:,1]
    if(len(mask_size)>2):
        mask1 = mask[:,:,0]
    else:
        mask1 = mask[:,:]
    mask2 = np.asanyarray(conv_mask_gt(mask1))
    mask2 = np.moveaxis(mask2, 1, 0)
    mask2 = np.moveaxis(mask2, 2, 1)
    
    batch_y_val[i,:,:,0] = mask2[:,:,0]

###########  For CopyMove Folder  ##########################
for i in range(test_batch_size):
    name = imgDirTest2[i]
    imgFilename = img_path2 + name
    img = imread(imgFilename)
    maskFilename = mask_path2 + name[:-7] + 'mask.png'
    mask = imread(maskFilename)
    
    img = imresize(img, [256,256,3])
    mask = imresize(mask, [256, 256])
    
    mask_size = mask.shape
    
    
    batch_x_val[test_batch_size+i,:,:,:] = img[:,:,0:3]
    batch_x1_val[test_batch_size+i,:,:,0] = img[:,:,1]
    if(len(mask_size)>2):
        mask1 = mask[:,:,0]
    else:
        mask1 = mask[:,:]
    mask2 = np.asanyarray(conv_mask_gt(mask1))
    mask2 = np.moveaxis(mask2, 1, 0)
    mask2 = np.moveaxis(mask2, 2, 1)
    
    batch_y_val[test_batch_size+i,:,:,0] = mask2[:,:,0]
    

best_loss = 100
best_acc = 0

file_acc = open('./file_dice_acc_2_decoder.txt', 'a')
file_loss = open('./file_dice_loss_2_decoder.txt', 'a')

file_acc_test = open('./file_dice_acc_test_2_decoder.txt', 'a')
file_loss_test = open('./file_dice_loss_test_2_decoder.txt', 'a')

for iters in range(num_iters):
    
    print('Epoch count: ', iters)
    
    random.shuffle(imgDirTrain1)
    random.shuffle(imgDirTrain2)
    
    ###########  For Splice Folder  ##########################
    for b in range(batch_size):
        name = imgDirTrain1[b]
        print('Image Name: ', name)
        imgFilename = img_path1 + name
        img = imread(imgFilename)
        maskFilename = mask_path1 + name[:-7] + 'mask.png'
        mask = imread(maskFilename)
        
        img = imresize(img, [256,256,3])
        mask = imresize(mask, [256, 256])
        
        mask_size = mask.shape
        
        
        batch_x[b,:,:,:] = img[:,:,0:3]
        batch_x1[b,:,:,0] = img[:,:,1]
        if(len(mask_size)>2):
            mask1 = mask[:,:,0]
        else:
            mask1 = mask[:,:]
        mask2 = np.asanyarray(conv_mask_gt(mask1))
        mask2 = np.moveaxis(mask2, 1, 0)
        mask2 = np.moveaxis(mask2, 2, 1)
        
        batch_y[b,:,:,0] = mask2[:,:,0]

    ###########  For CopyMove Folder  ##########################
    for b in range(batch_size):
        name = imgDirTrain2[b]
        print('Image Name: ', name)
        imgFilename = img_path2 + name
        img = imread(imgFilename)
        maskFilename = mask_path2 + name[:-7] + 'mask.png'
        mask = imread(maskFilename)
        
        img = imresize(img, [256,256,3])
        mask = imresize(mask, [256, 256])
        
        mask_size = mask.shape
        
        
        batch_x[batch_size+b,:,:,:] = img[:,:,0:3]
        batch_x1[batch_size+b,:,:,0] = img[:,:,1]
        if(len(mask_size)>2):
            mask1 = mask[:,:,0]
        else:
            mask1 = mask[:,:]
        mask2 = np.asanyarray(conv_mask_gt(mask1))
        mask2 = np.moveaxis(mask2, 1, 0)
        mask2 = np.moveaxis(mask2, 2, 1)
        
        batch_y[batch_size+b,:,:,0] = mask2[:,:,0]
        
            
            
    loss_train, acc_train = model.train_on_batch([batch_x1, batch_x], batch_y)
    
    ### constrain on the first layer of the Noise stream encoder ##############
    
    layer_output = model.get_layer('constrain').get_weights()
    layer_weight = layer_output[0]
    layer_bias = layer_output[1]
    
    constrained_weight = normalise(layer_weight)
    model.get_layer('constrain').set_weights([constrained_weight, layer_bias])
    
    ###########################################################################

    

    
    loss_test, acc_test = model.test_on_batch([batch_x1_val, batch_x_val], batch_y_val)
    
    print('Train loss: ', loss_train, ' acc: ', acc_train, ' Test loss: ', loss_test,  ' acc: ', acc_test)
    
    loss_train_str = str(loss_train) + '\n'
    acc_train_str = str(acc_train) + '\n'
   
    loss_train_str_test = str(loss_test) + '\n'
    acc_train_str_test = str(acc_test) + '\n'
    
    file_acc.write(acc_train_str)
    file_loss.write(loss_train_str)

    file_acc_test.write(acc_train_str_test)
    file_loss_test.write(loss_train_str_test)
    
    if(loss_train <=best_loss):
        print('Model with best loss found..')
        weight_path = './2_Decoder_models/Dice/Synthetic_NIST_models/loss/model_weights.hd5'
        model.save_weights(weight_path)
        best_loss = loss_train
    
    if(acc_train >= best_acc):
        print('Model with best acc found..')
        weight_path = './2_Decoder_models/Dice/Synthetic_NIST_models/acc/model_weights.hd5'
        model.save_weights(weight_path)
        best_acc = acc_train
            
        
file_acc.close()
file_loss.close()

file_acc_test.close()
file_loss_test.close()

print('.........Training is Done .............')


