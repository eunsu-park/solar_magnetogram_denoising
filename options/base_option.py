import argparse
import os


class BaseOption():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--seed", type=int, default=2331,
            help="random seed")

        self.parser.add_argument("--name_data", type=str, default="los_45",
            help="name of data, los_45, los_720, los, inclination, azimuth, disambig, field, vector_r, vector_t, vector_p")
        
        self.parser.add_argument("--gpu_ids", type=str, default="0",
            help="gpu id, ex) 0,2,3")
        self.parser.add_argument("--ch_inp", type=int, default=1,
            help="number of input channel")
        self.parser.add_argument("--ch_tar", type=int, default=1,
            help="number of target channel")

        self.parser.add_argument("--nb_down", type=int, default=8,
            help="# of down samplings oin UNet")

        self.parser.add_argument("--root_data", type=str,
            default=os.path.join("/data/eunsu/Dataset/denoising"),
            help="path to load data")
        self.parser.add_argument("--root_save", type=str,
            default=os.path.join("/data/eunsu/Result/denoising"),
            help="path to save result")

    def parse(self):

        opt = self.parser.parse_args()

        if opt.name_data == "los_45" :
            opt.path_data = "%s/los_45" % (opt.root_data)
            opt.pattern = "*.los_45.npz"
            opt.minmax = 1000.

        elif opt.name_data == "los_720" :
            opt.path_data = "%s/los_720" % (opt.root_data)
            opt.pattern = "*.los_720.npz"
            opt.minmax = 1000.

        elif opt.name_data == "los" :
            opt.path_data = "%s/los_*" % (opt.root_data)
            opt.pattern = "*.los_*.npz"
            opt.minmax = 1000.

        elif opt.name_data == "inclination" :
            opt.path_data = "%s/inclination" % (opt.root_data)
            opt.pattern = "*.inclination.npz"
            opt.minmax = 1000.

        elif opt.name_data == "azimuth" :
            opt.path_data = "%s/azimuth" % (opt.root_data)
            opt.pattern = "*.azimuth.npz"
            opt.minmax = 1000.

        elif opt.name_data == "disambig" :
            opt.path_data = "%s/disambig" % (opt.root_data)
            opt.pattern = "*.disambig.npz"
            opt.minmax = 1000.

        elif opt.name_data == "field" :
            opt.path_data = "%s/field" % (opt.root_data)
            opt.pattern = "*.field.npz"
            opt.minmax = 1000.

        elif opt.name_data == "vector_r" :
            opt.path_data = "%s/Vector" % (opt.root_data)
            opt.pattern = "*vector_r.npy"
            opt.minmax = 1000.

        elif opt.name_data == "vector_t" :
            opt.path_data = "%s/Vector" % (opt.root_data)
            opt.pattern = "*.sav"
            opt.minmax = 1000.

        elif opt.name_data == "vector_p" :
            opt.path_data = "%s/Vector" % (opt.root_data)
            opt.pattern = "*.sav"
            opt.minmax = 1000.

        return opt
