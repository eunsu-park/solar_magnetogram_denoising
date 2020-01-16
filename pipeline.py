import numpy as np
from random import shuffle
from utils_data import bytescale

class BaseData:
    def __init__(self, isize, lim_hmi):
        self.isize = isize
        self.lim_hmi = lim_hmi
    def make_tensor(self, file_):
        x = np.load(file_)[None, :, :, None]/self.lim_hmi
        return x
    def make_output(self, gen_):
        x = gen_.numpy().reshape(self.isize, self.isize)*self.lim_hmi
        x_png = bytescale(x, imin=-30, imax=30)
        return x, x_png
            
class train_batch_generator(BaseData):
    def __init__(self, list_train, isize, bsize, lim_hmi):
        super(train_batch_generator, self).__init__(isize=isize, lim_hmi=lim_hmi)
        self.list_train = list_train
        self.nb_train = len(self.list_train)
        self.size = bsize
        self.epoch = 0
        self.i = 0
    def __next__(self):
        while True:
            if self.i + self.size > self.nb_train :
                shuffle(self.list_train)
                self.i = 0
                self.epoch += 1
            batch_A = np.concatenate([self.make_tensor(self.list_train[j][0]) for j in range(self.i, self.i+self.size)], 0)
            batch_B = np.concatenate([self.make_tensor(self.list_train[j][1]) for j in range(self.i, self.i+self.size)], 0)
            self.i += self.size
            return self.epoch, batch_A, batch_B
        
class test_batch_generator(BaseData):
    def __init__(self, list_test, isize, lim_hmi):
        super(test_batch_generator, self).__init__(isize=isize, lim_hmi=lim_hmi)
        self.list_test = list_test
        self.nb_test = len(self.list_test)
        self.size = 1
        self.i = 0
    def __next__(self):
        while True:
            if self.i + self.size > self.nb_test :
                self.i = 0
            file_test = self.list_test[self.i]
            batch_A = self.make_tensor(file_test)
            date = file_test.split('.')[-2]
            self.i += self.size
            return date, batch_A