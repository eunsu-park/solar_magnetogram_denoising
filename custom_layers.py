import tensorflow as tf
import tensorflow.keras as keras

class conv_upsample(keras.layers.Layer):
    def __init__(self, filters, size=2, interp='nearest'):
        super(conv_upsample, self).__init__()
        if interp.lower() not in ['bilinear', 'nearest'] :
            raise NameError('interp: bilinear or nearest')
        self.interp = interp.lower()
        if type(size) == int :
            size = (size, size)
        self.size = size
        self.filters = filters
        self.up = keras.layers.UpSampling2D(size = self.size, interpolation=self.interp)
        self.conv = keras.layers.Conv2D(filters=filters,
                                        kernel_size=3,
                                        strides=1,
                                        padding='valid',
                                        kernel_initializer=init_conv,
                                        use_bias=False)
    def output_shape(self, shape):
        return (shape[0], shape[1]*self.size[0], shape[2]*self.size[1], self.filters)
    def __call__(self, x):
        layer = self.up (x)
        layer = keras.layers.ZeroPadding2D(1) (layer)
        layer = self.conv(layer)
        return layer
    
class reflect_padding(keras.layers.Layer):
    def __init__(self, padding=(1, 1)):
        super(reflect_padding, self).__init__()
        if type(padding) == int :
            padding = (padding, padding)
        self.padding = padding
    def output_shape(self, shape):
        return (shape[0], shape[1] + 2 * self.padding[0], shape[2] + 2 * self.padding[1], shape[3])
    def __call__(self, x):
        w_pad, h_pad = self.padding	
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')
    
    
class block_down(keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, use_norm=True):
        super(block_down, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_norm = use_norm
        self.conv = keras.layers.Conv2D(filters=self.filters,
                                        kernel_size=self.kernel_size,
                                        strides=self.strides,
                                        padding='valid',
                                        kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                        use_bias = not self.use_norm)
        self.norm = keras.layers.BatchNormalization(momentum=0.9, axis=-1,
                                                    epsilon=1.01e-5,
                                                    gamma_initializer=tf.random_normal_initializer(1., 0.02))
    def __call__(self, inp):
        if self.padding > 0 :
            layer = keras.layers.ZeroPadding2D(self.padding) (inp)
        layer = self.conv(layer)
        if self.use_norm:
            layer = self.norm(layer)
        return layer
        

class block_up(keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, cropping, use_norm=True):
        super(block_up, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.cropping = cropping
        self.use_norm = use_norm
        self.conv = keras.layers.Conv2DTranspose(filters=self.filters,
                                                 kernel_size=self.kernel_size,
                                                 strides=self.strides,
                                                 padding='valid',
                                                 kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                                 use_bias = not self.use_norm)
        self.norm = keras.layers.BatchNormalization(momentum=0.9, axis=-1,
                                                    epsilon=1.01e-5,
                                                    gamma_initializer=tf.random_normal_initializer(1., 0.02))
    def __call__(self, inp):
        layer = self.conv(inp)
        if self.cropping > 0 :
            layer = keras.layers.Cropping2D(self.cropping) (layer)
        if self.use_norm:
            layer = self.norm(layer)           
            
        return layer

class block_residual(keras.layers.Layer):
    def __init__(self, filters):
        super(block_residual, self).__init__()
        self.conv1 = keras.layers.Conv2D(filters=filters,
                                         kernel_size=3,
                                         strides=1,
                                         padding='valid',
                                         kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                         use_bias=False)
        self.norm1 = keras.layers.BatchNormalization(momentum=0.9,
                                                     axis=-1,
                                                     epsilon=1.01e-5,
                                                     gamma_initializer=tf.random_normal_initializer(1., 0.02))
        self.conv2 = keras.layers.Conv2D(filters=filters,
                                         kernel_size=3,
                                         strides=1,
                                         padding='valid',
                                         kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                         use_bias=False)
        self.norm2 = keras.layers.BatchNormalization(momentum=0.9,
                                                     axis=-1,
                                                     epsilon=1.01e-5,
                                                     gamma_initializer=tf.random_normal_initializer(1., 0.02))
    def __call__(self, x):
        layer = keras.layers.ZeroPadding2D(1) (x)
        layer = self.conv1(layer)
        layer = self.norm1(layer)
        layer = keras.layers.Activation('relu') (layer)
        layer = keras.layers.ZeroPadding2D(1) (layer)
        layer = self.conv2(layer)
        layer = self.norm2(layer)
        layer = keras.layers.Add() ([x, layer])
        return layer
        
        
        
        
        
        
        
        
        
        
        
        
        
        