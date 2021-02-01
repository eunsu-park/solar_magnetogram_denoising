class OptionBase:
    def __init__(self):
        self.gpu_id = 1
        self.display_frequency = 5000
        self.ckpt_frequency = self.display_frequency // 10

        self.ch_input = 1
        self.ch_output = 1
        self.lim_hmi = 100
        self.isize = 256

        self.root_data = '/userhome/park_e/datasets/denoising'
        self.root_save = '/mnt/storage225/park_e/results/tensorflow2/denoising'

class OptionTrainGAN(OptionBase):
    def __init__(self):
        super(OptionTrainGAN, self).__init__()
        
        self.is_train = True

        self.bsize = 32

        self.epoch_max = 500
        
        self.use_l1_loss = True
        self.weight_l1 = 100.
        
        self.use_fm_loss = False
        self.weight_fm = 10.

        self.learning_rate = 0.0002
        self.decay_epoch = 10
        self.decay_rate = 0.96
        self.staircase = False
        
        self.beta_1 = 0.5
        self.beta_2 = 0.999
        self.epsilon = 1e-8
        
        self.path_data = '%s/train'%(self.root_data)
        self.path_save = '%s/gan_%d'%(self.root_save, self.lim_hmi)
        
        
class OptionTrainAE(OptionBase):
    def __init__(self):
        super(OptionTrainAE, self).__init__()
        
        self.is_train = True

        self.bsize = 32

        self.epoch_max = 50
        
        self.learning_rate = 0.0002
        self.decay_epoch = 1
        self.decay_rate = 0.96
        self.staircase = False
        
        self.beta_1 = 0.5
        self.beta_2 = 0.999
        self.epsilon = 1e-8
        
        self.path_data = '%s/ae'%(self.root_data)
        self.path_save = '%s/ae_%d'%(self.root_save, self.lim_hmi)

class OptionTest(OptionBase):
    def __init__(self):
        super(OptionTest, self).__init__()

        self.is_train = False
        self.epoch = 500
