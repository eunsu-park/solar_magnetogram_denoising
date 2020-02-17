import tensorflow as tf
from options import OptionTrainGAN
from pipelines import DatasetGAN, DataLoaderGAN, Normalize
from imageio import imsave
import numpy as np
import os, time, warnings
warnings.filterwarnings('ignore')

opt = OptionTrainGAN()

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[opt.gpu_id], 'GPU')
tf.config.experimental.set_memory_growth(gpus[opt.gpu_id], True)

dataset_train = DatasetGAN('%s/*.npy'%(opt.path_data), transforms=[Normalize(opt.lim_hmi)])
dataloader_train = DataLoaderGAN(dataset_train, bsize=opt.bsize, shuffle=True)

nb_train = len(dataset_train)
nb_batch = len(dataloader_train)

print(nb_train, nb_batch)

path_model = '%s/model'%(opt.path_save)
path_snap = '%s/snap'%(opt.path_save)
os.makedirs(path_model, exist_ok=True)
os.makedirs(path_snap, exist_ok=True)

from networks import patch_discriminator, shallow_generator

from networks import unet_generator, shallow_generator, patch_discriminator
network_D = patch_discriminator(opt.isize, opt.ch_input, opt.ch_output)
network_G = shallow_generator(opt.isize, opt.ch_input, opt.ch_output)
network_G.summary()
network_D.summary()

loss_function_gan = tf.keras.losses.BinaryCrossentropy()
loss_function_mae = tf.keras.losses.MeanAbsoluteError()

decay_steps = opt.decay_epoch * nb_batch
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(opt.learning_rate,
                                                             decay_steps=decay_steps,
                                                             decay_rate=opt.decay_rate,
                                                             staircase=opt.staircase)
optimizer_D = tf.keras.optimizers.Adam(opt.learning_rate,
                                       beta_1=opt.beta_1,
                                       beta_2=opt.beta_2,
                                       epsilon=opt.epsilon)
optimizer_G = tf.keras.optimizers.Adam(lr_schedule,
                                       beta_1=opt.beta_1,
                                       beta_2=opt.beta_2,
                                       epsilon=opt.epsilon)

def loss_function_D(output_D_real, output_D_fake):
    loss_D_real = loss_function_gan(tf.ones_like(output_D_real[0], dtype=tf.float32), output_D_real[0])
    loss_D_fake = loss_function_gan(tf.zeros_like(output_D_fake[0], dtype=tf.float32), output_D_fake[0])
    loss_D = (loss_D_real + loss_D_fake)/2.
    return loss_D

def loss_function_G(real_B, fake_B, output_D_real, output_D_fake, weight_l1=opt.weight_l1, weight_fm=opt.weight_fm):
    loss_G_fake = loss_function_gan(tf.ones_like(output_D_fake[0], dtype=tf.float32), output_D_fake[0])

    feature_real = output_D_real[1:]
    feature_fake = output_D_fake[1:]

    loss_F = 0
    for i in range(len(feature_fake)):
        loss_F += loss_function_mae(feature_real[i], feature_fake[i])
    loss_F *= weight_fm
    
    loss_L = loss_function_mae(real_B, fake_B) * weight_l1

    return loss_G_fake, loss_F, loss_L

@tf.function
def train_step(real_A, real_B):

    with tf.GradientTape() as tape_G, tf.GradientTape() as tape_D:

        fake_B = network_G(real_A, training=True)
        
        output_D_real = network_D([real_A, real_B], training=True)
        output_D_fake = network_D([real_A, fake_B], training=True)
        
        loss_D = loss_function_D(output_D_real, output_D_fake)

        loss_G_fake, loss_F, loss_L = loss_function_G(real_B, fake_B, output_D_real, output_D_fake)
        
        loss_G = loss_G_fake

        if opt.use_l1_loss == True :
            loss_G = loss_G + loss_L

        if opt.use_fm_loss == True :
            loss_G = loss_G + loss_F

    gradient_G = tape_G.gradient(loss_G, network_G.trainable_variables)
    gradient_D = tape_D.gradient(loss_D, network_D.trainable_variables)

    optimizer_G.apply_gradients(zip(gradient_G, network_G.trainable_variables))
    optimizer_D.apply_gradients(zip(gradient_D, network_D.trainable_variables))

    return loss_D, loss_G_fake, loss_F, loss_L

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
loss_D_sum = loss_G_sum = loss_F_sum = loss_L_sum = 0
loss_D_mean = loss_G_mean = loss_F_mean = loss_L_mean = 0

epoch_max = opt.epoch_max
display_frequency = opt.display_frequency
ckpt_frequency = opt.ckpt_frequency

while epoch < epoch_max :
    
    for _ in range(len(dataloader_train)):
        i, inp, tar = next(dataloader_train)

        loss_D, loss_G, loss_F, loss_L = train_step(inp, tar)
        
        loss_D_sum += loss_D
        loss_G_sum += loss_G
        loss_F_sum += loss_F
        loss_L_sum += loss_L
        
        iter_ += 1
        
        if iter_ % display_frequency == 0 :

            loss_D_mean = loss_D_sum / display_frequency
            loss_G_mean = loss_G_sum / display_frequency
            loss_F_mean = loss_F_sum / display_frequency
            loss_L_mean = loss_L_sum / display_frequency

            message = (epoch, epoch_max, iter_,
                       loss_D_mean, loss_G_mean, loss_F_mean, loss_L_mean, int(time.time()-t0))
            message = '[%d/%d][%d] Loss_D: %5.3f Loss_G: %5.3f Loss_F: %5.3f Loss_L: %5.3f Time: %dsec'%message
            print(message)
            
            loss_D_sum = loss_G_sum = loss_F_sum = loss_L_sum = 0
            loss_D_mean = loss_G_mean = loss_F_mean = loss_L_mean = 0
            
            snap_inp = inp[0:1]
            snap_tar = tar[0:1]
            snap_gen = generation_step(snap_inp)
            snap_inp = snap_inp.numpy().reshape(opt.isize, opt.isize)
            snap_tar = snap_tar.numpy().reshape(opt.isize, opt.isize)
            snap_gen = snap_gen.numpy().reshape(opt.isize, opt.isize)
            snap = (np.hstack((snap_inp, snap_tar, snap_gen)) * opt.lim_hmi).clip(-15, 15)
            snap = ((snap + 15.) * (255./30.)).astype(np.uint8)
            imsave('%s/denoising.gan.%07d.png'%(path_snap, iter_), snap)
            
            t0 = time.time()
            
        if iter_ % ckpt_frequency == 0 :

            name_ = 'denoising.gan.last'
            network_G.save('%s/%s.G.h5'%(path_model, name_))
            network_D.save('%s/%s.D.h5'%(path_model, name_))



    epoch += 1
    
    name_ = 'denoising.gan.%04d'%(epoch)
    network_G.save('%s/%s.G.h5'%(path_model, name_))
    network_D.save('%s/%s.D.h5'%(path_model, name_))
        
            

            
        
        
        
        
        
        
        
        
        
        
