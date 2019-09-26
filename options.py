from utils import make_tensor, make_output

class option_base():
    
    def __init__(self):

        print('\n------------ Options for base ------------\n')
        
        name_models = ['UNET', 'SUNET', 'SSUNET', 'CSSUNET']
        print('\nAvailable Model: UNET, SUNET, SSUNET, CSSUNET\n')
        answer = input('Select Model?     ')
        if answer.upper() not in name_models :
            raise NameError('%s: Invalid model name'%(answer))
        self.name_model = answer.upper()
        del answer

        print('\n# of layer in PatchGAN Discriminator (Receptive Field Size): 0(1), 1(16), 2(34), 3(70), 4(142), 5(286)\n')
        answer = input('# of layers?     ')
        if int(answer) not in range(6):
            raise ValueError('%s: Invalid # of layers in Discriminator'%(answer))
        self.layer_max_d = int(answer)
        del answer
        
        self.isize = 256
        self.ch_axis = 1
        self.mode = '%s_%dD' % (self.name_model, self.layer_max_d)

        self.instr_input = 'hmi'
        self.instr_output = 'hmi'
        self.name_input = 'center'
        self.name_output = 'stacks'
        self.ch_input = 1
        self.ch_output = 1
        self.drange = 100.
        
        self.root_data = '/home/park_e/datasets'
        self.root_save = '/userhome/park_e/solar_magnetogram_denoising'
        
        self.root_ckpt = '%s/%s/ckpt'%(self.root_save, self.mode)
        self.root_snap = '%s/%s/snap'%(self.root_save, self.mode)
        self.root_test = '%s/%s/test'%(self.root_save, self.mode)
        
        self.make_tensor = make_tensor(self.isize, self.drange)
        self.make_output = make_output(self.isize, self.drange)
        
class option_train(option_base):
    
    def __init__(self):
        super(option_train, self).__init__()
        
        print('\n------------ Options for train ------------\n')
        
        self.gpu_id = input('GPU ID?     ')
        self.iter_display = int(input('Display frequency(iter)?     '))
        self.iter_save = int(input('Save frequency(iter)?     '))
        self.iter_max = int(input('Max iteration?     '))
        
        self.bsize = 1
