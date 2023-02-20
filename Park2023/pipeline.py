import torch
from torch.utils import data
import numpy as np
from glob import glob
from sunpy.map import Map
import torch.nn.functional as F
from torchvision.transforms import Compose
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def func(x, a, b, c):
    return a*np.exp(-(x-b)**2/(2*c**2))

class FitGaussian(object):
    def __call__(self, data):
        tmp = np.histogram(data.flatten(), bins=np.linspace(-30, 30, 61))
        x = np.linspace(-29.5, 29.5, 60)
        popt, _ = curve_fit(func, x, tmp[0])
        return abs(popt[2])

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
        if opt.is_train == True :
            pattern = '%s/train/*.fits'%(opt.root_data)
        else :
            pattern = '%s/test/*.fits'%(opt.root_data)
        self.list_data = glob(pattern)
        self.nb_data = len(self.list_data)

        self.read_fits = ReadFits()
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
        data = self.read_fits(self.list_data[idx])
        patch = self.random_crop(data)[None, :, :]
        noise = self.generate_gaussian(loc=0, scale=self.fit_gaussian(patch), size=patch.shape)
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


    opt = TrainOption().parse()

    dataset = GaussianDataset(opt, is_train=True)
    dataloader = data.DataLoader(dataset, batch_size=8, num_workers=16)
    print(len(dataset), len(dataloader))

    imgs = []

    # fig = plt.figure(figsize=(9, 3))
    # for idx, (patch, noise, gaussian) in enumerate(dataloader):
    #     patch = patch.numpy()
    #     noise = noise.numpy()
    #     gaussian = gaussian.numpy()
    #     print(idx, patch.dtype, noise.dtype, gaussian.dtype)

    #     patch = patch[0][0]
    #     noise = noise[0][0]
    #     gaussian = gaussian[0][0]

    #     img = np.hstack([patch, noise, gaussian])
    #     img = img * opt.minmax
    #     img = (img.copy() + 30.) * (255./60.)
    #     img = np.clip(img, 0, 255).astype(np.uint8)

    #     plot = plt.imshow(img, cmap='gray', animated=True)
    #     plt.tight_layout()
    #     imgs.append([plot])

    #     if idx == 100 :
    #         break

    # animate = animation.ArtistAnimation(fig, imgs, interval=500, blit=True)
    # animate.save('%s_data.gif' % (opt.prefix))
    # plt.close()