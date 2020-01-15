import tensorflow as tf
import tensorflow.keras as keras


conv_init=tf.random_normal_initializer(0., 0.02)
gamma_init=tf.random_normal_initializer(1., 0.02)


def down_conv(nb_feature, *a, **k):
    return keras.layers.Conv2D(filters=nb_feature, *a, **k, kernel_initializer=conv_init, use_bias=False)

def up_conv(nb_feature, *a, **k):
    return keras.layers.Conv2DTranspose(filters=nb_feature, *a, **k,
                                        kernel_initializer=conv_init, use_bias=False)

def batch_norm():
    return keras.layers.BatchNormalization(momentum=0.9, axis=-1, epsilon=1.01e-5, gamma_initializer=gamma_init)

def up_sample(*a, **k):
    return keras.layers.UpSampling2D(*a, **k)

def dropout(*a, **k):
    return keras.layers.Dropout(*a, **k)

def leaky_relu(*a, **k):
    return keras.layers.LeakyReLU(*a, **k)

def activation(*a, **k):
    return keras.layers.Activation(*a, **k)

def concatenate(*a, **k):
    return keras.layers.Concatenate(*a, **k)

def zero_pad(*a, **k):
    return keras.layers.ZeroPadding2D(*a, **k)

def crop(*a, **k):
    return keras.layers.Cropping2D(*a, **k)


class reflect_padding:
    def __init__(self, padding=(1,1)):
        if type(padding) == int:
            padding = (padding, padding)
        self.padding = padding

    def output_shape(self, shape):
        return (shape[0], shape[1] + 2 * self.padding[0], shape[2] + 2 * self.padding[1], shape[3])

    def __call__(self, x):
        w_pad, h_pad = self.padding	
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT') 


class upsample_convolution:
    def __init__(self, filters, size=2, interp='bilinear'):
        if interp.lower() not in ['bilinear', 'nearest'] :
            raise NameError('interp: bilinear or nearest')
        else :
            self.interp = interp.lower()
        if type(size) == int :
            size = (size, size)
        self.size = size
        self.filters = filters
    def output_shape(self, shape, size):
        return (shape[0], shape[1]*self.size[0], shape[2]*self.size[1], self.filters)
    def __call__(self, x):
        layer = up_sample(size = self.size, interpolation=self.interp) (x)
        layer = reflect_padding(1) (layer)
        layer = down_conv(self.filters, kernel_size=3, padding='valid') (layer)
        return layer

class block_encoder:
    def __init__(self, nb_feature, use_bn=True):
        self.nb_feature = nb_feature
        self.use_bn = use_bn
    def __call__(self, x):
        layer = leaky_relu(0.2) (layer)
        layer = reflect_padding(1) (x)
        layer = down_conv(self.nb_feature, kernel_size=4, strides=2, padding='valid') (layer)
        if self.use_bn :
            layer = batch_norm() (layer)
        return layer

class block_decoder:
    def __init__(self, nb_feature, use_bn=True):
        self.nb_feature = nb_feature
        self.use_bn = use_bn
    def __call__(self, x):
        layer = activation('relu') (x)
        layer = up_conv(self.nb_feature, kernel_size=4, strides=2, padding='valid') (layer)
        layer = crop(1) (layer)
        if self.use_bn :
            layer = batch_norm() (layer)
        return layer

class block_residual : 
    def __init__(self, nb_feature):
        self.nb_feature = nb_feature
    def __call__(self, x):
        layer = reflect_padding(1) (x)
        layer = down_conv(self.nb_feature, kernel_size=3, strides=1, padding='valid') (layer)
        layer = batch_norm() (layer)
        layer = activation('relu') (layer)
        layer = reflect_padding(1) (layer)
        layer = down_conv(self.nb_feature, kernel_size=3, strides=1, padding='valid') (layer)
        layer = batch_norm() (layer)
        layer = keras.layers.Add() ([x, layer])
        return layer


def generator(isize, ch_input, ch_output, nb_layer_g = 4, nb_feature_g=64, nb_feature_max=512, use_tanh=False):

    input_A=keras.layers.Input(shape=(isize, isize, ch_input), dtype=tf.float32)
    list_nb_feature=[]
    list_layer_encoder=[]

    nb_feature=min(nb_feature_g, nb_feature_max)
    list_nb_feature.append(nb_feature)
    layer = reflect_padding(1) (input_A)
    layer = down_conv(nb_feature, kernel_size=4, strides=2, padding='valid') (layer)
    list_layer_encoder.append(layer)

    for n in range(nb_layer_g):
        nb_feature=min(nb_feature*2, nb_feature_max)
        list_nb_feature.append(nb_feature)
        layer = block_encoder(nb_feature) (layer)
        list_layer_encoder.append(layer)

    for m in range(nb_layer_g*2):
        layer = block_residual(nb_feature) (layer)

    list_layer_encoder=list(reversed(list_layer_encoder))
    list_nb_feature=list(reversed(list_nb_feature[:-1]))

    for k in range(nb_layer_g):
        layer=concatenate(axis=-1) ([layer, list_layer_encoder[k]])
        nb_feature=list_nb_feature[k]
        layer=block_decoder(nb_feature) (layer)

    layer=concatenate (axis=-1)([layer, list_layer_encoder[-1]])
    layer=activation('relu') (layer)
    layer=up_conv(ch_output, kernel_size=4, strides=2, padding='valid') (layer)
    last_ = crop(1) (layer)    
    if use_tanh :
        last_=activation('tanh') (last_)
    
    return keras.models.Model(inputs=[input_A], outputs=[last_])


def discriminator(isize, ch_input, ch_output, nb_layer_d=3, nb_feature_d=64, nb_feature_max=512):

    input_A=keras.layers.Input(shape=(isize, isize, ch_input), dtype=tf.float32, name='input')
    output_B=keras.layers.Input(shape=(isize, isize, ch_output), dtype=tf.float32, name='target')
    nb_feature=nb_feature_d
    
    features = []
    
    layer=concatenate(-1)([input_A, output_B])
    
    if nb_layer_d==0:
        layer=down_conv(64, kernel_size=1, strides=1, padding='same') (layer)
        layer=leaky_relu(0.2) (layer)
        features.append(layer)
        layer=down_conv(128, kernel_size=1, strides=1, padding='same') (layer)
        layer=batch_norm() (layer)
        layer=leaky_relu(0.2) (layer)
        features.append(layer)
        layer=down_conv(1, kernel_size=1, strides=1, padding='same') (layer)
        last_=activation('sigmoid') (layer)

    else :
        for i in range(nb_layer_d):
            layer=down_conv(nb_feature, kernel_size=4, strides=2, padding='same') (layer)
            if i > 0 :
                layer=batch_norm() (layer)
            layer=leaky_relu(0.2) (layer)
            features.append(layer)
            nb_feature=min(nb_feature*2, nb_feature_max)

        layer=zero_pad(1) (layer)
        layer=down_conv(nb_feature, kernel_size=4, strides=1) (layer)
        layer=batch_norm() (layer)
        layer=leaky_relu(0.2) (layer)
        features.append(layer)
        layer=zero_pad(1) (layer)
        layer=down_conv(1, kernel_size=4, strides=1) (layer)
        last_=activation('sigmoid') (layer)

    return keras.models.Model(inputs=[input_A, output_B], outputs=[last_]+features)

if __name__ == '__main__' :
    M = generator(256, 1, 1)
    M.summary()
