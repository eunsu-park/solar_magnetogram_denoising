import tensorflow as tf
import numpy as np
from glob import glob
from random import shuffle

class DataLoader:
    def __init__(self, dataset, bsize, shuffle=True):
        self.dataset = dataset
        self.bsize = bsize
        self.i = 0
        self.nb_data = len(self.dataset)
        self.shuffle = shuffle
        self.nb_batch = len(self.dataset)//self.bsize
        if len(self.dataset)%self.bsize != 0 :
            self.nb_batch += 1
        
    def __len__(self):
        return self.nb_batch
    
    def __next__(self):
        if self.i + 1 > self.nb_batch :
            self.i = 0
            if self.shuffle==True :
                self.dataset.Shuffle()

        idx_start = (self.i)*self.bsize
        idx_end = (self.i+1)*self.bsize if (self.i+1)*self.bsize <= self.nb_data else self.nb_data

        inps = []
        tars = []

        for j in range(idx_start, idx_end):
            inp, tar = self.dataset[j]

            inps.append(inp)
            tars.append(tar)

        batch_A = np.concatenate([inps[k] for k in range(len(inps))], 0)
        batch_B = np.concatenate([tars[k] for k in range(len(tars))], 0)
        self.i += 1

        batch_A = tf.cast(batch_A, tf.float32)
        batch_B = tf.cast(batch_B, tf.float32)

        return self.i, batch_A, batch_B


class DatasetAE:
    def __init__(self, pattern, transforms=None):
        self.list_data = glob(pattern)
        self.transforms = transforms
        self.nb_data = len(self.list_data)
        self.AddNoise = AddGaussian(0, 10)

    def __len__(self):
        return self.nb_data
    
    def Shuffle(self):
        shuffle(self.list_data)

    def __getitem__(self, idx):
        file_ = self.list_data[idx]
        img = np.load(file_)[None,:,:,None]
        inp, tar = self.AddNoise(img)
        if self.transforms :
            for n in range(len(self.transforms)):
                inp = self.transforms[n](inp)
                tar = self.transforms[n](tar)
        return inp, tar


class DatasetGAN:
    def __init__(self, pattern, transforms=None):
        self.list_data = glob(pattern)
        self.transforms = transforms
        self.nb_data = len(self.list_data)
        
    def __len__(self):
        return self.nb_data
    
    def Shuffle(self):
        shuffle(self.list_data)
        
    def __getitem__(self, idx):
        file_ = self.list_data[idx]
        img = np.load(file_)[None,:,:,:]
        inp = img[:,:,:,0:1]
        tar = img[:,:,:,1:2]
        if self.transforms :
            for n in range(len(self.transforms)):
                inp = self.transforms[n](inp)
                tar = self.transforms[n](tar)
        return inp, tar
        
                 
class Normalize:
    def __init__(self, lim_hmi=100):
        self.lim_hmi = float(lim_hmi)
    def __call__(self, img):
        img = (img/self.lim_hmi).clip(-1., 1.)
        return img

    
class ToTensor:
    def __init__(self, dtype=tf.float32):
        self.dtype = dtype
    def __call__(self, img):
        return tf.cast(img, self.dtype)

                 
class AddGaussian:
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma
    def __call__(self, image):
        dif_mean = np.random.randint(201)/100 - 1.
        dif_sigma = np.random.randint(801)/100 - 4.
        noise = np.random.normal(self.mean+dif_mean, self.sigma+dif_sigma, image.shape)
        noisy = image.copy() + noise
        return noisy, image

    
class AddSNP:
    def __init__(self, percentage, ratio_sp=0.5):
        self.percentage = percentage
        self.ratio_sp = ratio_sp
    def __call__(self, image):
        num_s = np.ceil(self.percentage*image.size*self.ratio_sp)
        num_p = np.ceil(self.percentage*image.size*(1-self.ratio_sp))
        coord_s =  [np.random.randint(0, i - 1, int(num_s)) for i in image.shape]
        coord_p =  [np.random.randint(0, i - 1, int(num_p)) for i in image.shape]
        noisy = image.copy()
        noisy[coord_s] = 1
        noisy[coord_p] = -1
        return noisy


class AddPoissin:
    def __call__(self, image):
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poissin(image*vals) / float(vals)
        return noisy


class AddSpeckle:
    def __call__(self, image):
        gauss = np.random.randn(image.shape)
        noisy = image + image * gauss
        return noisy                 

                 
class CropPatches:
    def __init__(self, isize, bsize):
        self.isize = isize
        self.bsize = bsize
    def __call__(self, image):
        tmp = []
        for i in range(self.bsize):
            x = np.random.randint(2048)+1024
            y = np.random.randint(2048)+1024
            patch = image[:, x-(self.isize//2):x+(self.isize//2),
                          y-(self.isize//2):y+(self.isize//2), :]
            tmp.append(patch)
        cropped = np.concatenate([tmp[n] for n in range(self.bsize)], 0)
        return cropped
