import torch
from torch.utils import data
import numpy as np
from glob import glob
from sunpy.map import Map
import torch.nn.functional as F
from torchvision.transforms import Compose
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.io import readsav


def func(x, a, b, c):
    return a*np.exp(-(x-b)**2/(2*c**2))


class FitGaussian(object):
    def __init__(self, minmax=150):
        self.minmax=minmax
    def __call__(self, data):
        tmp = np.histogram(data.flatten(), bins=np.linspace(-self.minmax, self.minmax, self.minmax+1))
        x = np.linspace(-self.minmax+0.5, self.minmax-0.5, self.minmax)
        popt, _ = curve_fit(func, x, tmp[0])
        amp = popt[0]
        loc = popt[1]
        scale = abs(popt[2])
        return amp, loc, scale


class FitMultiGaussian(object):
    def __init__(self, minmax=150):
        self.minmax=minmax

    def fit_negative(self, data_):
        w = np.where(data_ <= 0.)
        data = data_[w]
        tmp = np.histogram(data.flatten(), bins=np.linspace(-self.minmax, self.minmax, self.minmax+1))
        x = np.linspace(-self.minmax+0.5, self.minmax-0.5, self.minmax)
        popt, _ = curve_fit(func, x, tmp[0])
        amp = popt[0]
        loc = popt[1]
        scale = abs(popt[2])
        return amp, loc, scale

    def fit_positive(self, data_):
        w = np.where(data_ >= 0.)
        data = data_[w]
        tmp = np.histogram(data.flatten(), bins=np.linspace(-self.minmax, self.minmax, self.minmax+1))
        x = np.linspace(-self.minmax+0.5, self.minmax-0.5, self.minmax)
        popt, _ = curve_fit(func, x, tmp[0])
        amp = popt[0]
        loc = popt[1]
        scale = abs(popt[2])
        return amp, loc, scale

    def __call__(self, data):
        popt_negative = self.fit_negative(data)
        popt_positive = self.fit_positive(data)
        return popt_negative, popt_positive


class GenerateGaussian(object):
    def __call__(self, loc, scale, size):
        return np.random.normal(loc=loc, scale=scale, size=size)

class Normalize(object):
    def __init__(self, minmax):
        self.minmax = minmax
    def __call__(self, data):
        return data/self.minmax

class ReadFits(object):
    def __call__(self, fits):
        return Map(fits).data.astype(np.float64)
    
class ReadSav(object):
    def __init__(self, name_data):
        self.name_data = name_data
    def __call__(self, file_sav):
        sav = readsav(file_sav)
        return sav[self.name_data].copy()
        

class Cast(object):
    def __call__(self, data):
        return torch.from_numpy(data.astype(np.float32))

class RandomCrop(object):
    def __init__(self, patch_size):
        self.patch_size = patch_size
    def __call__(self, data):
        x, y = np.random.randint(2048-512, 2048+512-self.patch_size, 2)
        return data[x:x+self.patch_size, y:y+self.patch_size]
    
class BaseDataset(data.Dataset):
    def __init__(self, opt):
        if opt.name_data in ["M_45s", "M_720s"] :
            ext = "fits"
            self.reader = ReadFits()
            pattern = '%s/%s'%(opt.root_data, opt.name_data)
            self.fit_minmax = 30.
        elif opt.name_data in ["BT", "BR", "BP"] :
            ext = "sav"
            self.reader = ReadSav(opt.name_data)
            pattern = '%s/B_720s'%(opt.root_data)
            self.fit_minmax = 100
        print(ext)
        if opt.is_train == True :
            pattern = '%s/train/*.%s'%(pattern, ext)
        else :
            pattern = '%s/%s/*.%s'%(pattern, ext)
        self.list_data = glob(pattern)
        self.nb_data = len(self.list_data)

        self.random_crop = RandomCrop(opt.patch_size)
        self.compose = Compose([Normalize(opt.minmax), Cast()])
        
    def __len__(self):
        return self.nb_data



class GaussianDataset(BaseDataset):
    def __init__(self, opt):
        super(GaussianDataset, self).__init__(opt)
        self.fit_gaussian = FitGaussian()
        self.generate_gaussian = GenerateGaussian()

    def __getitem__(self, idx):
        data = self.reader(self.list_data[idx])
        patch = self.random_crop(data)[None, :, :]
        popt = self.fit_gaussian(patch)
        noise = self.generate_gaussian(loc=popt[1], scale=popt[2], size=patch.shape)
        gaussian = patch.copy() + noise.copy()
        
        patch = self.compose(patch)
        noise = self.compose(noise)
        gaussian = self.compose(gaussian)

        return patch, noise, gaussian

class MultiGaussianDataset(BaseDataset):
    def __init__(self, opt):
        super(MultiGaussianDataset, self).__init__(opt)
        self.fit_gaussian = FitMultiGaussian()
        self.generate_gaussian = GenerateGaussian()

    def __getitem__(self, idx):
        data = self.reader(self.list_data[idx])
        patch = self.random_crop(data)[None, :, :]
        popt_negative, popt_positive = self.fit_gaussian(patch)
        noise_negative = self.generate_gaussian(
            loc=popt_negative[1], scale=popt_negative[2], size=patch.shape)
        noise_positive = self.generate_gaussian(
            loc=popt_positive[1], scale=popt_positive[2], size=patch.shape)
        noise = noise_negative + noise_positive
        gaussian = patch.copy() + noise.copy()
        
        patch = self.compose(patch)
        noise = self.compose(noise)
        gaussian = self.compose(gaussian)

        return patch, noise, gaussian



if __name__ == "__main__" :

    from options.train_option import TrainOption
    from imageio import imsave
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from utils.others import define_dataset_and_model


    opt = TrainOption().parse()
    dataloader, network = define_dataset_and_model(opt)

    imgs = []

    fig = plt.figure(figsize=(9, 3))
    for idx, (patch, noise, gaussian) in enumerate(dataloader):
        patch = patch.numpy()
        noise = noise.numpy()
        gaussian = gaussian.numpy()
        print(idx, patch.dtype, noise.dtype, gaussian.dtype)

        patch = patch[0][0]
        noise = noise[0][0]
        gaussian = gaussian[0][0]

        img = np.hstack([patch, noise, gaussian])
        img = img * opt.minmax
        img = (img.copy() + 30.) * (255./60.)
        img = np.clip(img, 0, 255).astype(np.uint8)

        plot = plt.imshow(img, cmap='gray', animated=True)
        plt.tight_layout()
        imgs.append([plot])

        if idx == 100 :
            break

    animate = animation.ArtistAnimation(fig, imgs, interval=500, blit=True)
    animate.save('%s_data.gif' % (opt.name_data))
    plt.close()