import argparse


class BaseOption():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--prefix', type=str, default='denoising')
        self.parser.add_argument('--seed', type=int, default=1220)

        self.parser.add_argument('--crop_size', type=int, default=256)
        self.parser.add_argument('--ch_inp', type=int, default=1)
        self.parser.add_argument('--ch_tar', type=int, default=1)

        self.parser.add_argument('--nb_feat_init', type=int, default=64)
        self.parser.add_argument('--nb_feat_max', type=int, default=512)
        self.parser.add_argument('--nb_down', type=int, default=6)
        self.parser.add_argument('--use_tanh', type=bool, default=False)

        self.parser.add_argument('--root_data', type=str, default='/userhome/park_e/datasets/denoising')
        self.parser.add_argument('--root_save', type=str, default='/userhome/park_e/results/tf_denoising')

    def parse(self):
        return self.parser.parse_args()

class TrainOption(BaseOption):
    def __init__(self):
        super(TrainOption, self).__init__()

        self.parser.add_argument('--is_train', type=bool, default=True)
        self.parser.add_argument('--epoch_max', type=int, default=200)
        self.parser.add_argument('--report_frequency', type=int, default=5)
        self.parser.add_argument('--save_frequency', type=int, default=10)

        self.parser.add_argument('--gpu_id', type=int, default=0)

        self.parser.add_argument('--batch_size', type=int, default=8)
        self.parser.add_argument('--workers', type=int, default=4)

        self.parser.add_argument('--lr_init', type=float, default=0.001)

        self.parser.add_argument('--decay_steps', type=int, default=1000)
        self.parser.add_argument('--decay_rate', type=float, default=0.96)

class TestOption(BaseOption):
    def __init__(self):
        super(TestOption, self).__init__()

        self.parser.add_argument('--is_train', type=bool, default=False)
        self.parser.add_argument('--gpu_id', type=int, default=3)
        self.parser.add_argument('--batch_size', type=int, default=8)
        self.parser.add_argument('--epoch', type=int, default=200)
