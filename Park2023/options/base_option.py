import argparse


class BaseOption():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--seed", type=int, default=2331,
            help="random seed")
        self.parser.add_argument("--gpu_ids", type=str, default="0",
            help="gpu id, ex) 0,2,3")
        self.parser.add_argument("--ch_inp", type=int, default=1,
            help="number of input channel")
        self.parser.add_argument("--ch_tar", type=int, default=1,
            help="number of target channel")
        self.parser.add_argument("--minmax", type=float, default=1000,
            help="data normalization factor")
        self.parser.add_argument("--root_data", type=str,
            default="/DATA/eunsu/denoising/dataset",
            help="path to load data")
        self.parser.add_argument("--root_save", type=str,
            default="/DATA/eunsu/denoising/result",
            help="path to save result")

    def parse(self):
        return self.parser.parse_args()