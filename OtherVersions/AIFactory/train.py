from option import TrainOption
opt = TrainOption().parse()

import tensorflow as tf
import numpy as np
import random

tf.random.set_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)

import os, time, warnings
warnings.filterwarnings('ignore')

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[opt.gpu_id], 'GPU')
tf.config.experimental.set_memory_growth(gpus[opt.gpu_id], True)

path_model = '%s/model'%(opt.root_save)
path_snap = '%s/snap'%(opt.root_save)

if not os.path.exists(path_model):
    os.makedirs(path_model)
if not os.path.exists(path_snap):
    os.makedirs(path_snap)

from pipeline import DeNormalizer, ImageMaker, CustomSequence
from imageio import imsave

denorm = DeNormalizer()
image_maker = ImageMaker()
sequence = CustomSequence(opt, phase='train')
sequence_test = CustomSequence(opt, phase='test')

nb_train = len(sequence)
nb_test = len(sequence_test)

from network import Generator

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=opt.lr_init,
    decay_steps=opt.decay_steps,
    decay_rate=opt.decay_rate, staircase=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

network = Generator(opt)        
network.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

print(nb_train, nb_test)


epoch = 0
while epoch < opt.epoch_max :

    print('From epoch %d to %d'%(epoch+1, epoch+opt.report_frequency))

    network.fit(sequence,
                epochs=opt.report_frequency, workers=opt.workers)
    epoch += opt.report_frequency
    A = network.evaluate(sequence_test, verbose=0)
    mse = A[0]
    mae = A[1]
    print('Epoch:%d MSE:%6.3f MAE:%6.3f' %(epoch, mse, mae))

    idx = np.random.randint(0, nb_test)
    inp, tar = sequence_test[idx]
    gen = network.predict(inp)

    snap_inp = np.hstack([inp[n,:,:,0] for n in range(4)])*1000.
    snap_tar = np.hstack([tar[n,:,:,0] for n in range(4)])*1000.
    snap_gen = np.hstack([gen[n,:,:,0] for n in range(4)])*1000.
    snap = np.vstack([snap_inp, snap_tar, snap_gen])
    snap = image_maker(snap)
    imsave('%s/epoch_%04d.png'%(path_snap, epoch), snap)

    if epoch % opt.save_frequency == 0 :
        network.save('%s/epoch_%04d.h5'%(path_model, epoch))
