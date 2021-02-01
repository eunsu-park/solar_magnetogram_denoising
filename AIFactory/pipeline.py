import tensorflow as tf
import numpy as np
from glob import glob
from random import shuffle


class Normalizer:
    def __call__(self, data):
        return np.clip(data, -3000., 3000.)/1000.

class DeNormalizer :
    def __call__(self, data):
        return np.clip(data*1000., -3000., 3000.)

class Caster :
    def __call__(self, data):
        return tf.cast(data, tf.float32)

class ImageMaker:
    def __call__(self, data):
        return np.clip((data+30.)*(255./60.), 0, 255).astype(np.uint8)

class CustomSequence(tf.keras.utils.Sequence):
    def __init__(self, opt, phase='train'):
        self.phase = phase
        if self.phase == 'train' :
            pattern = '%s/train/*.npz'%(opt.root_data)
        else :
            pattern = '%s/test/*.npz'%(opt.root_data)
        self.list_data = sorted(glob(pattern))
        self.nb_data = len(self.list_data)
        self.batch_size = opt.batch_size
        self.nb_batch = int(np.ceil(self.nb_data/self.batch_size))
        self.norm = Normalizer()
        self.cast = Caster()
        self.indexs = [n for n in range(self.nb_data)]

    def __loadFile__(self, file_):
        npz = np.load(file_)
        inp = npz['inp'][None,12:-12,12:-12,None]
        tar = npz['tar'][None,12:-12,12:-12,None]
        return inp, tar

    def __getBatch__(self, indexs):
        inps = []
        tars = []
        for index in indexs :
            inp, tar = self.__loadFile__(self.list_data[index])
            if self.phase == 'train' :
                choice = np.random.rand()
                if choice >= 0.5 :
                    inps.append(inp)
                    tars.append(tar)
                else :
                    noise = np.random.normal(0., 10., inp.shape)
                    inps.append(inp + noise)
                    tars.append(inp)
            else :
                inps.append(inp)
                tars.append(tar)
        inps = np.concatenate(inps, 0)
        tars = np.concatenate(tars, 0)
        return inps, tars

    def __len__(self):
        return self.nb_batch

    def one_epoch_end(self):
        if self.phase == 'train' :
            np.random.shuffle(self.indexs)

    def __getitem__(self, idx):
        idx_start = idx*self.batch_size
        idx_end = idx_start + self.batch_size
        indexs = self.indexs[idx_start:idx_end]
        inps, tars = self.__getBatch__(indexs)

        inps = self.norm(inps)
        tars = self.norm(tars)
        inps = self.cast(inps)
        tars = self.cast(tars)
        
        return inps, tars

if __name__ == '__main__' :

    from option import TrainOption
    opt = TrainOption().parse()
    sequence = CustomSequence(opt)
    print(len(sequence))

    epochs = 10

    for epoch in range(epochs):
        for idx, (inp, tar) in enumerate(sequence):
            print(epoch, idx, inp.shape, np.min(inp), np.max(inp), tar.shape, np.min(tar), np.max(tar))
