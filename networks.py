import tensorflow as tf
import tensorflow.keras as keras
from math import log2
from custom_layers import block_down, block_up, block_residual

def patch_discriminator(isize, ch_input, ch_output, nb_downsampling=0,
                  nb_layer_d=3, nb_feature_d=64, nb_feature_max=512):

    input_A = keras.layers.Input(shape=(isize, isize, ch_input),
                                 dtype=tf.float32, name='input')
    input_B = keras.layers.Input(shape=(isize, isize, ch_output),
                                 dtype=tf.float32, name='target')
    nb_feature = nb_feature_d
    features = []
    layer = keras.layers.Concatenate(-1) ([input_A, input_B])
    
    for d in range(nb_downsampling):
        layer = keras.layers.AveragePooling2D(pool_size=3, strides=2, padding='same') (layer)
    
    if nb_layer_d == 0 :
        layer = block_down(filters=64, kernel_size=1, strides=1, padding=0, use_norm=False) (layer)
        layer = keras.layers.LeakyReLU(0.2) (layer)
        features.append(layer)
        layer = block_down(filters=128, kernel_size=1, strides=1, padding=0, use_norm=True) (layer)
        layer = keras.layers.LeakyReLU(0.2) (layer)
        features.append(layer)
        layer = block_down(filters=1, kernel_size=1, strides=1, padding=0, use_norm=False) (layer)
        
    else :
        layer = block_down(filters=nb_feature, kernel_size=4, strides=2, padding=1, use_norm=False) (layer)
        layer = keras.layers.LeakyReLU(0.2) (layer)
        features.append(layer)
        nb_feature = min(nb_feature*2, nb_feature_max)
        
        for i in range(1, nb_layer_d):
            layer = block_down(filters=nb_feature, kernel_size=4, strides=2, padding=1, use_norm=True) (layer)
            layer = keras.layers.LeakyReLU(0.2) (layer)
            features.append(layer)
            nb_feature = min(nb_feature*2, nb_feature_max)
            
        layer = block_down(filters=nb_feature, kernel_size=4, strides=1, padding=1, use_norm=True) (layer)
        layer = keras.layers.LeakyReLU(0.2) (layer)
        features.append(layer)
        
        layer = block_down(filters=1, kernel_size=4, strides=1, padding=1, use_norm=False) (layer)
        
    layer = keras.layers.Activation('sigmoid') (layer)
    return keras.models.Model(inputs = [input_A, input_B], outputs = [layer] + features, name='discriminator')
        
def unet_generator(isize, ch_input, ch_output,
                   nb_feature_g=64, nb_feature_max=512, use_tanh=False):

    input_A = keras.layers.Input(shape=(isize, isize, ch_input), dtype=tf.float32)
    list_nb_feature = []
    list_layer_encoder = []
    nb_block = int(log2(isize)) - 2

    layer = input_A
    nb_feature = nb_feature_g
    list_nb_feature.append(nb_feature)
    layer = block_down(filters=nb_feature, kernel_size=4, strides=2, padding=1, use_norm=False) (layer)
    list_layer_encoder.append(layer)

    for e in range(nb_block):
        nb_feature = min(nb_feature*2, nb_feature_max)
        list_nb_feature.append(nb_feature)
        layer = keras.layers.LeakyReLU(0.2) (layer)
        layer = block_down(filters=nb_feature, kernel_size=4, strides=2, padding=1, use_norm=True) (layer)
        list_layer_encoder.append(layer)
        
    nb_feature = min(nb_feature*2, nb_feature_max)
        
    layer = keras.layers.LeakyReLU(0.2) (layer)
    layer = block_down(filters=nb_feature, kernel_size=4, strides=2, padding=1, use_norm=False) (layer)
    layer = keras.layers.Activation('relu') (layer)
    layer = block_up(filters=nb_feature, kernel_size=4, strides=2, cropping=1, use_norm=True) (layer)
    layer = keras.layers.Dropout(0.5) (layer)

    list_layer_encoder = list(reversed(list_layer_encoder))
    list_nb_feature = list(reversed(list_nb_feature[:-1]))
    
    for d in range(nb_block):
        layer = keras.layers.Concatenate(axis=-1) ([layer, list_layer_encoder[d]])
        nb_feature = list_nb_feature[d]
        layer = block_up(filters=nb_feature, kernel_size=4, strides=2, cropping=1, use_norm=True) (layer)
        if d < 2 :
            layer = keras.layers.Dropout(0.5) (layer)
            
    layer = keras.layers.Concatenate(axis=-1) ([layer, list_layer_encoder[-1]])
    layer = keras.layers.Activation('relu') (layer)
    layer = block_up(filters=ch_output, kernel_size=4, strides=2, cropping=1, use_norm=False) (layer)
    if use_tanh == True :
        layer = keras.layers.Activation('tanh') (layer)

    return keras.models.Model(inputs = [input_A], outputs = [layer], name='generator')


def shallow_generator(isize, ch_input, ch_output, nb_layer_g=4,
                       nb_feature_g=64, nb_feature_max=512, use_tanh=False):

    input_A = keras.layers.Input(shape=(isize, isize, ch_input), dtype=tf.float32)
    list_nb_feature = []
    list_layer_encoder = []

    layer = input_A
    nb_feature = nb_feature_g
    list_nb_feature.append(nb_feature)
    layer = block_down(filters=nb_feature, kernel_size=4, strides=2, padding=1, use_norm=False) (layer)
    list_layer_encoder.append(layer)

    for e in range(nb_layer_g):
        layer = keras.layers.LeakyReLU(0.2) (layer)
        nb_feature = min(nb_feature*2, nb_feature_max)
        list_nb_feature.append(nb_feature)
        layer = block_down(filters=nb_feature, kernel_size=4, strides=2, padding=1, use_norm=True) (layer)
        list_layer_encoder.append(layer)
        
    nb_feature = min(nb_feature*2, nb_feature_max)
    
    for r in range(9):
        layer = block_residual(nb_feature) (layer)
        
    list_layer_encoder = list(reversed(list_layer_encoder))
    list_nb_feature = list(reversed(list_nb_feature[:-1]))
    
    for d in range(nb_layer_g):
        layer = keras.layers.Concatenate(axis=-1) ([layer, list_layer_encoder[d]])
        nb_feature = list_nb_feature[d]
        layer = keras.layers.Activation('relu') (layer)
        layer = block_up(filters=nb_feature, kernel_size=4, strides=2, cropping=1, use_norm=True) (layer)
            
    layer = keras.layers.Concatenate(axis=-1) ([layer, list_layer_encoder[-1]])
    layer = keras.layers.Activation('relu') (layer)
    layer = block_up(filters=ch_output, kernel_size=4, strides=2, cropping=1, use_norm=False) (layer)

    if use_tanh == True :
        layer = keras.layers.Activation('tanh') (layer)

    return keras.models.Model(inputs = [input_A], outputs = [layer], name='generator')
