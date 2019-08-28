import os
os.environ['KERAS_BACKEND']        = 'tensorflow'
os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'

import numpy as np
import glob
from imageio import imsave
import tensorflow as tf
import keras.backend as K
from keras.optimizers import Adam
import time
from random import shuffle


from options import option_train
class train(option_train):
    
    def __init__(self):
        super(train, self).__init__()
        
    def build_network(self):
        from networks import PATCH_DISCRIMINATOR as network_D
        self.network_D = network_D(self.isize, self.ch_input, self.ch_output, self.layer_max_d)
            
        if self.name_model == 'UNET' :
            from networks import UNET as network_G
        elif self.name_model == 'SUNET' :
            from networks import SUNET as network_G
        elif self.name_model == 'SSUNET' :
            from networks import SSUNET as network_G
        elif self.name_model == 'CSSUNET' :
            from networks import SUNET as network_G
        self.network_G = network_G(self.isize, self.ch_input, self.ch_output)

    def make_directory(self):
        
        os.makedirs(self.root_ckpt, exist_ok=True)
        os.makedirs(self.root_snap, exist_ok=True)        
        os.makedirs(self.root_test, exist_ok=True)        

    def make_dataset(self):
        
        path_train_input = '%s/train/%s/%s'%(self.root_data, self.instr_input, self.name_input)
        path_train_output = '%s/train/%s/%s'%(self.root_data, self.instr_output, self.name_output)
        
        list_train_input = sorted(glob.glob('%s/*/*/*/*.npy'%path_train_input))
        list_train_output = sorted(glob.glob('%s/*/*/*/*.npy'%path_train_output))
        
        nb_train_input = len(list_train_input)
        nb_train_output = len(list_train_output)
        
        if nb_train_input != nb_train_output :
            raise RuntimeError('# of train input(%d) and output(%d) are differnent'%(nb_train_input, nb_train_output))
        self.list_train = list(zip(list_train_input, list_train_output))
        self.nb_train = nb_train_input
        
        path_validation_input = '%s/validation/%s/%s'%(self.root_data, self.instr_input, self.name_input)
        list_validation_input = sorted(glob.glob('%s/*/*/*/*.npy'%path_validation_input))
        nb_validation_input = len(list_validation_input)
        self.list_validation = list_validation_input
        self.nb_validation = nb_validation_input
        
        path_test_input = '%s/test/%s/%s'%(self.root_data, self.instr_input, self.name_input)
        list_test_input = sorted(glob.glob('%s/*/*/*/*.npy'%path_test_input))
        nb_test_input = len(list_test_input)
        self.list_test = list_test_input
        self.nb_test = nb_test_input

    def data_loader(self):
        data_train_A = np.array([], dtype=np.float32)
        data_train_B = np.array([], dtype=np.float32)

        
    def train_batch_generator(self):
        data_AB = self.list_train
        length = self.nb_train
        epoch = i = 0
        tmpsz = None
        while True :
            sz = tmpsz if tmpsz else self.bsize
            if i + sz > length :
                shuffle(data_AB)
                i = 0
                epoch += 1
            batch_A = np.array([])
            batch_B = np.array([])
            for j in range(i, i+sz):
                data_A = self.make_tensor(data_AB[j][0])
                data_B = self.make_tensor(data_AB[j][1])
                batch_A = np.append(batch_A, data_A)
                batch_B = np.append(batch_B, data_B)
            batch_A.shape = (self.bsize, self.isize, self.isize, self.ch_input)
            batch_B.shape = (self.bsize, self.isize, self.isize, self.ch_output)
            i += sz
            tmpsz = yield epoch, batch_A.astype(np.float32), batch_B.astype(np.float32)       
            
            
    def run_validation(self):
        
        path_snap = '%s/iter_%07d'%(self.root_snap, self.iter_gen)
        os.makedirs(path_snap, exist_ok=True)
        
        for file_A in self.list_validation :
            real_A = self.make_tensor(file_A)
            fake_B = self.network_G.predict(real_A)
            fake_B, fake_B_png = self.make_output(fake_B)
            
            name_save = 'Denoised.%s.%s'%(self.mode, file_A.split('/')[-1][-23:-4])
            np.save('%s/%s.npy'%(path_snap, name_save), fake_B)
            imsave('%s/%s.png'%(path_snap, name_save), fake_B_png)
            
        print('Validation snaps (%d images) are saved in %s'%(self.nb_validation, path_snap))

    def run_test(self):
        
        path_test = '%s/iter_%07d'%(self.root_test, self.iter_gen)
        os.makedirs(path_test, exist_ok=True)
        
        for file_A in self.list_test :
            real_A = self.make_tensor(file_A)
            fake_B = self.network_G.predict(real_A)
            fake_B, fake_B_png = self.make_output(fake_B)
            
            name_save = 'Denoised.%s.%s'%(self.mode, file_A.split('/')[-1][-23:-4])
            np.save('%s/%s.npy'%(path_test, name_save), fake_B)
            imsave('%s/%s.png'%(path_test, name_save), fake_B_png)
            
        print('Test results (%d images) are saved in %s'%(self.nb_test, path_test))            
            
            
    def run(self):

        self.make_directory()
        self.make_dataset()
        train_batch_generator = self.train_batch_generator()        

        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_id
        K.set_image_data_format('channels_last')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.Session(config = config)

        self.build_network()

        real_A = self.network_G.input
        fake_B = self.network_G.output
        real_B = self.network_D.inputs[1]

        output_D_real = self.network_D([real_A, real_B])
        output_D_fake = self.network_D([real_A, fake_B])

        loss_FN = lambda output, target : -K.mean(K.log(output+1e-12)*target+K.log(1-output+1e-12)*(1-target))

        loss_D_real = loss_FN(output_D_real, K.ones_like(output_D_real))
        loss_D_fake = loss_FN(output_D_fake, K.zeros_like(output_D_fake))
        loss_G_fake = loss_FN(output_D_fake, K.ones_like(output_D_fake))

        loss_L = K.mean(K.abs(fake_B-real_B))

        loss_D = loss_D_real + loss_D_fake
        training_updates_D = Adam(lr = 2e-4, beta_1 = 0.5).get_updates(self.network_D.trainable_weights, [], loss_D)
        network_D_train = K.function([real_A, real_B], [loss_D/2.0], training_updates_D)

        loss_G = loss_G_fake + 100 * loss_L
        training_updates_G = Adam(lr = 2e-4, beta_1 = 0.5).get_updates(self.network_G.trainable_weights, [], loss_G)
        network_G_train = K.function([real_A, real_B], [loss_G_fake, loss_L], training_updates_G)

        t0 = time.time()
        t1 = time.time()
        self.iter_gen, epoch = 0, 0
        err_L, err_G, err_D = 0, 0, 0
        err_L_sum, err_G_sum, err_D_sum = 0, 0, 0
        
        print('\n--------------------------------\n')

        print('\nNow start below session!\n')
        print('Mode: %s'%self.mode)        
        print('Checkpoint save path: %s'%(self.root_ckpt))
        print('Validation snap save path: %s'%(self.root_snap))
        print('Test result save path: %s'%(self.root_test))        
        print('# of train, validation, and test datasets : %d, %d, %d'%(self.nb_train, self.nb_validation, self.nb_test))
        
        print('\n--------------------------------\n')        

        while self.iter_gen <= self.iter_max :

            epoch, train_A, train_B = next(train_batch_generator)
    
            err_G, err_L = network_G_train([train_A, train_B])
            err_D, = network_D_train([train_A, train_B])
    
            err_D_sum += err_D
            err_G_sum += err_G
            err_L_sum += err_L
    
            self.iter_gen += self.bsize
    
            if self.iter_gen % self.iter_display == 0:
        
                err_D_mean = err_D_sum/self.iter_display
                err_G_mean = err_G_sum/self.iter_display
                err_L_mean = err_L_sum/self.iter_display
        
                print('[%d][%d/%d] LOSS_D: %5.3f LOSS_G: %5.3f LOSS_L: %5.3f T: %dsec/%dits, Total T: %d'
                % (epoch, self.iter_gen, self.iter_max,
                   err_D_mean, err_G_mean, err_L_mean,
                   time.time()-t1, self.iter_display, time.time()-t0))
        
                err_L_sum, err_G_sum, err_D_sum = 0, 0, 0
        
                t1 = time.time()
        
            if self.iter_gen % self.iter_save == 0:
        
                dst_model_G = '%s/%s_iter_%07d_G.h5'%(self.root_ckpt, self.mode, self.iter_gen)
                dst_model_D = '%s/%s_iter_%07d_D.h5'%(self.root_ckpt, self.mode, self.iter_gen)
        
                self.network_G.save(dst_model_G)
                self.network_D.save(dst_model_D)
        
                print('network_G and network_D are saved under %s'%(self.root_ckpt))
        
                self.run_validation()
                self.run_test()
        
                t1 = time.time()    
        
if __name__ == '__main__' :
    do_train = train()
    do_train.run()            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
