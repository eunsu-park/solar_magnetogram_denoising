import tensorflow as tf
from options import OptionTrainAE
from pipelines import DatasetAE, DataLoader, Normalize
from imageio import imsave
import numpy as np
import os, time, warnings
warnings.filterwarnings('ignore')

opt = OptionTrainAE()

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[opt.gpu_id], 'GPU')
tf.config.experimental.set_memory_growth(gpus[opt.gpu_id], True)

dataset_train = DatasetAE('%s/*.npy'%(opt.path_data), transforms=[Normalize(opt.lim_hmi)])
dataloader_train = DataLoader(dataset_train, bsize=opt.bsize)

nb_train = len(dataset_train)
nb_batch = len(dataloader_train)

print(nb_train, nb_batch)

path_model = '%s/model'%(opt.path_save)
path_snap = '%s/snap'%(opt.path_save)
os.makedirs(path_model, exist_ok=True)
os.makedirs(path_snap, exist_ok=True)

from networks import shallow_generator

network_G = shallow_generator(opt.isize, opt.ch_input, opt.ch_output)
network_G.summary()

loss_function_mae = tf.keras.losses.MeanAbsoluteError()
loss_function_mse = tf.keras.losses.MeanSquaredError()

decay_steps = opt.decay_epoch * nb_batch
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(opt.learning_rate,
                                                             decay_steps=decay_steps,
                                                             decay_rate=opt.decay_rate,
                                                             staircase=opt.staircase)
optimizer_G = tf.keras.optimizers.Adam(lr_schedule,
                                       beta_1=opt.beta_1,
                                       beta_2=opt.beta_2,
                                       epsilon=opt.epsilon)

@tf.function
def train_step(real_A, real_B):

    with tf.GradientTape() as tape_G, tf.GradientTape() as tape_D:

        fake_B = network_G(real_A, training=True)
        
        mae = loss_function_mae(real_B, fake_B)
        mse = loss_function_mse(real_B, fake_B)
        
        loss_G = mse*1000.
        
    gradient_G = tape_G.gradient(loss_G, network_G.trainable_variables)
    optimizer_G.apply_gradients(zip(gradient_G, network_G.trainable_variables))

    return loss_G, mae, mse

@tf.function
def generation_step(real_A):
    return network_G(real_A, training=False)

def t_now():
    TM = time.localtime(time.time())
    return '%04d-%02d-%02d %02d:%02d:%02d'%(TM.tm_year, TM.tm_mon, TM.tm_mday, TM.tm_hour, TM.tm_min, TM.tm_sec)

print('\n------------------------------------ Summary ------------------------------------\n')

print('\n%s: Now start below session!\n'%(t_now()))
print('Model save path: %s'%(path_model))
print('Snap save path: %s'%(path_snap))
print('# of train and train batch: %d, %d'%(nb_train, nb_batch))

print('\n---------------------------------------------------------------------------------\n')

t0 = time.time()
epoch = iter_ = 0
loss_sum = mae_sum = mse_sum = 0
loss_mean = mae_mean = mse_mean = 0

epoch_max = opt.epoch_max
display_frequency = opt.display_frequency
ckpt_frequency = opt.ckpt_frequency

while epoch < epoch_max :
    
    for _ in range(len(dataloader_train)):
        i, inp, tar = next(dataloader_train)

        loss, mae, mse = train_step(inp, tar)
        
        loss_sum += loss
        mae_sum += mae
        mse_sum += mse
        
        iter_ += 1
        
        if iter_ % display_frequency == 0 :

            loss_mean = loss_sum / display_frequency
            mae_mean = mae_sum / display_frequency
            mse_mean = mse_sum / display_frequency

            message = (epoch, epoch_max, iter_,
                       loss_mean, mae_mean, mse_mean, int(time.time()-t0))
            message = '[%d/%d][%d] Loss: %5.3f MAE: %5.3f MSE: %5.3f Time: %dsec'%message
            print(message)
            
            loss_sum = mae_sum = mse_sum = 0
            loss_mean = mae_mean = mse_mean = 0
            
            snap_inp = inp[0:1]
            snap_tar = tar[0:1]
            snap_gen = generation_step(snap_inp)
            snap_app = generation_step(snap_tar)
            snap_inp = snap_inp.numpy().reshape(opt.isize, opt.isize)
            snap_tar = snap_tar.numpy().reshape(opt.isize, opt.isize)
            snap_gen = snap_gen.numpy().reshape(opt.isize, opt.isize)
            snap_app = snap_app.numpy().reshape(opt.isize, opt.isize)
            snap = (np.hstack((snap_inp, snap_tar, snap_gen, snap_app)) * opt.lim_hmi).clip(-15, 15)
            snap = ((snap + 15.) * (255./30.)).astype(np.uint8)
            imsave('%s/denoising.ae.%07d.png'%(path_snap, iter_), snap)
            
            t0 = time.time()
            
        if iter_ % ckpt_frequency == 0 :

            name_ = 'denoising.ae.last'
            network_G.save('%s/%s.G.h5'%(path_model, name_))


    epoch += 1
    
    name_ = 'denoising.ae.%04d'%(epoch)
    network_G.save('%s/%s.G.h5'%(path_model, name_))        
            

            
        
        
        
        
        
        
        
        
        
        
