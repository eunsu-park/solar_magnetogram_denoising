from option import TestOption
opt = TestOption().parse()

import tensorflow as tf
import numpy as np
import os, time, warnings
warnings.filterwarnings('ignore')

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[opt.gpu_id], 'GPU')
tf.config.experimental.set_memory_growth(gpus[opt.gpu_id], True)

path_model = '%s/model'%(opt.root_save)
path_gen = '%s/generation/epoch_%04d'%(opt.root_save, opt.epoch)
if not os.path.exists(path_gen):
    os.makedirs(path_gen)

network = tf.keras.models.load_model('%s/epoch_%04d.h5'%(path_model, opt.epoch), compile=False)

from pipeline import DeNormalizer, ImageMaker, CustomSequence

denorm = DeNormalizer()
sequence = CustomSequence(opt, phase='test')
list_data = sequence.list_data

nb_data = len(list_data)
nb_test = len(sequence)

print(nb_data, nb_test)

tars = []
gens = []

for _, (inp, tar) in enumerate(sequence):
    gen = network.predict(inp)
    tar = denorm(tar)
    gen = denorm(gen)
    tars.append(tar)
    gens.append(gen)

tars = np.concatenate(tars, 0)
gens = np.concatenate(gens, 0)

mae = tf.keras.metrics.MeanAbsoluteError()(tars, gens)
mse = tf.keras.metrics.MeanSquaredError()(tars, gens)
psnr = tf.image.psnr(tars+3000., gens+3000., 6000.)

print(mae, mse, np.mean(psnr))

print(tars.shape)
print(gens.shape)

for i in range(tars.shape[0]):
    name = list_data[i].split('/')[-1]
    tar = tars[i,:,:,0]
    gen = gens[i,:,:,0]

    np.savez('/home/park_e/tmp/test/%s'%(name), tar)
    np.savez('%s/%s'%(path_gen, name), gen)

