import keras

conv_init = keras.initializers.RandomNormal(0, 0.02)
gamma_init = keras.initializers.RandomNormal(1, 0.02)

def DOWN_CONV(nb_feature, *a, **k):
    return keras.layers.Conv2D(filters=nb_feature, *a, **k, kernel_initializer=conv_init, use_bias=False)

def UP_CONV(nb_feature, *a, **k):
    return keras.layers.Conv2DTranspose(filters=nb_feature, *a, **k, kernel_initializer=conv_init, use_bias=False)

def BATCH_NORM():
    return keras.layers.BatchNormalization(momentum=0.9, axis=-1, epsilon=1.01e-5, gamma_initializer = gamma_init)

def UP_SAMPLE(*a, **k):
    return keras.layers.UpSampling2D(*a, **k)

def DROPOUT(*a, **k):
    return keras.layers.Dropout(*a, **k)

def ACTIVATION(*a, **k):
    return keras.layers.Activation(*a, **k)

def LEAKY_RELU(*a, **k):
    return keras.layers.advanced_activations.LeakyReLU(*a, **k)

def CONCATENATE(*a, **k):
    return keras.layers.Concatenate(*a, **k)

def DENSE(*a, **k):
    return keras.layers.Dense(*a, **k)

def BLOCK_ENCODER(layer, nb_feature):
    layer = LEAKY_RELU(0.2) (layer)
    layer = DOWN_CONV(nb_feature, kernel_size=4, strides=2, padding='same') (layer)
    layer = BATCH_NORM() (layer, training=1)
    return layer

def BLOCK_INTMISS(layer, nb_feature):
    layer = LEAKY_RELU(0.2) (layer)
    layer = DOWN_CONV(nb_feature, kernel_size=4, strides=2, padding='same') (layer)
    layer = ACTIVATION('relu') (layer)
    layer = UP_CONV(nb_feature, kernel_size=4, strides=2, padding='same') (layer)
    layer = BATCH_NORM() (layer, training=1)
    layer = DROPOUT(0.5) (layer, training=1)
    return layer

def BLOCK_DECODER(layer, nb_feature):
    layer = ACTIVATION('relu') (layer)
    layer = UP_CONV(nb_feature, kernel_size=4, strides=2, padding='same') (layer)
    layer = BATCH_NORM() (layer, training=1)
    return layer

def UNET(isize, ch_input, ch_output, nb_feature_g=64, nb_feature_max=512, use_tanh=False):
    
    input_A = keras.layers.Input(shape=(isize, isize, ch_input), dtype='float32')
    current_size=isize
    
    list_nb_feature = []
    list_layer_encoder = []
    
    nb_feature = min(nb_feature_g, nb_feature_max)
    list_nb_feature.append(nb_feature)
    list_layer_encoder.append(DOWN_CONV(nb_feature, kernel_size=4, strides=2, padding='same') (input_A))
    current_size //= 2
        
    nb_block = 0
    while current_size != 2 :
        
        nb_feature = min(nb_feature*2, nb_feature_max)
        list_nb_feature.append(nb_feature)
        list_layer_encoder.append(BLOCK_ENCODER(list_layer_encoder[-1], nb_feature))
        current_size //= 2
        nb_block += 1
            
    layer = BLOCK_INTMISS(list_layer_encoder[-1], nb_feature)

    list_layer_encoder = list(reversed(list_layer_encoder))
    list_n_fearue = list(reversed(list_nb_feature[:-1]))
    
    for n in range(nb_block):
        layer = CONCATENATE(axis=-1)([layer, list_layer_encoder[n]])
        nb_feature = list_nb_feature[n]
        layer = BLOCK_DECODER(layer, nb_feature)
        current_size *= 2
        if current_size <= 8 :
            layer = DROPOUT(0.5) (layer, training=1)
    
    layer = CONCATENATE(axis=-1)([layer, list_layer_encoder[-1]])
    layer = ACTIVATION('relu') (layer)
    layer = UP_CONV(ch_output, kernel_size=4, strides=2, padding='same') (layer)
    current_size *= 2
    if use_tanh:
        layer = ACTIVATION('tanh') (layer)
    
    return keras.models.Model(inputs = input_A, outputs = layer)

def PATCH_DISCRIMINATOR(isize, ch_input, ch_output, layer_max_d=3, nb_feature_d=64, nb_feature_max=512):

    input_A = keras.layers.Input(shape=(isize, isize, ch_input), dtype='float32')
    input_B = keras.layers.Input(shape=(isize, isize, ch_output), dtype='float32')
    nb_feature = nb_feature_d
    
    layer = CONCATENATE(-1)([input_A, input_B])
    
    if layer_max_d == 0:
        layer = DOWN_CONV(64, kernel_size = 1, strides = 1, padding = 'same') (layer)
        layer = LEAKY_RELU(0.2) (layer)
        layer = DOWN_CONV(128, kernel_size = 1, strides = 1, padding = 'same') (layer)
        layer = BATCH_NORM() (layer, training = 1)
        layer = LEAKY_RELU(0.2) (layer)
        layer = DOWN_CONV(1, kernel_size = 1, strides = 1, padding = 'same') (layer)
        layer = ACTIVATION('sigmoid') (layer)

    else :
        for i in range(layer_max_d):

            layer = DOWN_CONV(nb_feature, kernel_size = 4, strides = 2, padding = 'same') (layer)
            if i > 0 : layer = BATCH_NORM() (layer, training = 1)
            layer = LEAKY_RELU(0.2) (layer)
            nb_feature = min(nb_feature*2, nb_feature_max)

        layer = DOWN_CONV(nb_feature, kernel_size = 4, strides = 1, padding = 'same') (layer)
        layer = BATCH_NORM() (layer, training = 1)
        layer = LEAKY_RELU(0.2) (layer)

        layer = DOWN_CONV(1, kernel_size = 4, strides = 1, padding = 'same') (layer)
        layer = ACTIVATION('sigmoid') (layer)

    return keras.models.Model(inputs = [input_A, input_B], outputs = layer)
