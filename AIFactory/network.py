import tensorflow as tf

init_conv = tf.random_normal_initializer(0, 0.02)
init_batnorm = tf.random_normal_initializer(1., 0.02)

def Conv2D(kernel_initializer=init_conv, *a, **k):
    return tf.keras.layers.Conv2D(kernel_initializer=kernel_initializer, *a, **k)

def Conv2DTranspose(kernel_initializer=init_conv, *a, **k):
    return tf.keras.layers.Conv2DTranspose(kernel_initializer=kernel_initializer, *a, **k)

def BatchNormalization(gamma_initializer=init_batnorm):
    return tf.keras.layers.BatchNormalization(momentum=0.9, axis=-1, epsilon=1.01e-5,
                                              gamma_initializer=gamma_initializer)

def Input(*a, **k):
    return tf.keras.layers.Input(*a, **k)

def Concatenate(*a, **k):
    return tf.keras.layers.Concatenate(*a, **k)

def Cropping2D(size):
    return tf.keras.layers.Cropping2D(size)

def ZeroPadding2D(size):
    return tf.keras.layers.ZeroPadding2D(size)

def LeakyReLU(alpha):
    return tf.keras.layers.LeakyReLU(alpha)

def ReLU():
    return tf.keras.layers.Activation('relu')

def Tanh():
    return tf.keras.layers.Activation('tanh')

def Dropout(ratio):
    return tf.keras.layers.Dropout(ratio)

def Generator(opt):

    input_A = Input(shape=(None, None, opt.ch_inp), dtype=tf.float32)
    list_nb_feat = []
    list_layer = []

    nb_feat = opt.nb_feat_init
    list_nb_feat.append(nb_feat)
    layer = input_A
    size = opt.crop_size

    layer = ZeroPadding2D(1) (layer)
    layer = Conv2D(filters=nb_feat, kernel_size=4, strides=2, padding='valid', use_bias=True) (layer)
    size //= 2
    list_layer.append(layer)

    for e in range(opt.nb_down):
        nb_feat = min(nb_feat*2, opt.nb_feat_max)
        list_nb_feat.append(nb_feat)
        layer = LeakyReLU(0.2) (layer)
        layer = ZeroPadding2D(1) (layer)
        layer = Conv2D(filters=nb_feat, kernel_size=4, strides=2, padding='valid', use_bias=False) (layer)
        size //= 2
        layer = BatchNormalization() (layer)
        list_layer.append(layer)

    nb_feat = min(nb_feat*2, opt.nb_feat_max)

    layer = LeakyReLU(0.2) (layer)
    layer = ZeroPadding2D(1) (layer)
    layer = Conv2D(filters=nb_feat, kernel_size=4, strides=2, padding='valid', use_bias=True) (layer)
    size //= 2
    layer = ReLU() (layer)
    layer = Conv2DTranspose(filters=nb_feat, kernel_size=4, strides=2, padding='valid', use_bias=False) (layer)
    layer = Cropping2D(1) (layer)
    size *= 2
    layer = BatchNormalization() (layer)
    if size <= 8 :
        layer = Dropout(0.5) (layer)

    list_layer = list(reversed(list_layer))
    list_nb_feat = list(reversed(list_nb_feat[:-1]))

    for d in range(opt.nb_down):
        layer = Concatenate(-1) ([layer, list_layer[d]])
        nb_feat = list_nb_feat[d]
        layer = ReLU() (layer)
        layer = Conv2DTranspose(filters=nb_feat, kernel_size=4, strides=2, padding='valid', use_bias=False) (layer)
        layer = Cropping2D(1) (layer)
        size *= 2
        if size <= 8 :
            layer = Dropout(0.5) (layer)

    layer = Concatenate(-1) ([layer, list_layer[-1]])
    layer = ReLU() (layer)
    layer = Conv2DTranspose(filters=opt.ch_tar, kernel_size=4, strides=2, padding='valid', use_bias=True) (layer)
    layer = Cropping2D(1) (layer)
    size *= 2

    if opt.use_tanh == True :
        layer = Tanh() (layer)

    model = tf.keras.models.Model(inputs=[input_A], outputs=[layer])
    model.summary()
    return model

    
if __name__ == '__main__' :
    from option import TrainOption
    opt = TrainOption().parse()

    gpu_id = opt.gpu_id
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[gpu_id], True)

    inp = tf.zeros((opt.batch_size, opt.crop_size, opt.crop_size, opt.ch_inp), dtype=tf.float32)
    tar = tf.zeros((opt.batch_size, opt.crop_size, opt.crop_size, opt.ch_tar), dtype=tf.float32)

    network = Generator(opt)
    gen = network(inp)

    print(inp.shape, tar.shape, gen.shape)
