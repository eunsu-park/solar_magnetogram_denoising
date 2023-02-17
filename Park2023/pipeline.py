import torch
from torch.utils import data
import numpy as np
from glob import glob
from sunpy.map import Map
import torch.nn.functional as F
from torchvision.transforms import Compose


class GenerateGaussian(object):
    def __call__(self, loc, scale, size):
        return np.random.normal(loc=loc, scale=scale, size=size)

class GenerateStackedGaussian(object):
    def __init__(self, loc, scale, steps):
        self.generate_gaussian = GenerateGaussian(loc=loc, scale=scale)
        self.steps = steps
    def __call__(self, data):
        inp = data.copy()
        step = np.random.randint(self.steps) + 1
        for _ in range(step):
            tar = inp.copy()
            noise = self.generate_gaussian(tar)
            inp = tar.copy() + noise
        return inp, tar

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
    
class GenerateDiffusion(object):
    def __init__(self, beta_start, beta_end, steps):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.steps = steps
        self.build_diffusion()

    def build_diffusion(self):
        betas = np.linspace(self.beta_start, self.beta_end, self.steps)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        self.sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1. - alphas_cumprod)

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = np.take(a, t, -1)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def __call__(self, data, noise):
        t = np.random.randint(self.steps)
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, np.array([t]), data.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, np.array([t]), data.shape)
        return sqrt_alphas_cumprod_t * data + sqrt_one_minus_alphas_cumprod_t * noise


# class GaussianDataset(BaseDataset):
#     def __init__(self, opt):
#         super(GaussianDataset, self).__init__(opt)



class BaseDataset(data.Dataset):
    def __init__(self, opt):
        self.list_data = glob('%s/*.fits'%(opt.root_data))
        self.nb_data = len(self.list_data)

        self.read_fits = ReadFits()
        self.random_crop = RandomCrop(opt.patch_size)
        self.generate_gaussian = GenerateGaussian()
        self.generate_diffusion = GenerateDiffusion(beta_start=opt.beta_start, beta_end=opt.beta_end, steps=opt.steps)
        self.compose = Compose([Normalize(opt.minmax), Cast()])

    def __len__(self):
        return self.nb_data

    def __getitem__(self, idx):
        data = self.read_fits(self.list_data[idx])
        patch = self.random_crop(data)[None, :, :]
        loc = patch.mean()
        scale = patch.std()
        noise = self.generate_gaussian(loc, scale, patch.shape)
        gaussian = patch.copy() + noise.copy()
        diffusion = self.generate_diffusion(patch.copy(), noise.copy())
        
        patch = self.compose(patch)
        noise = self.compose(noise)
        gaussian = self.compose(gaussian)
        diffusion = self.compose(diffusion)

        return patch, noise, gaussian, diffusion







if __name__ == "__main__" :

    from options.train_option import TrainOption
    from imageio import imsave
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt


    opt = TrainOption().parse()

    dataset = BaseDataset(opt)
    dataloader = data.DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.num_workers)
    print(len(dataset), len(dataloader))

    imgs = []

    fig = plt.figure(figsize=(11, 3))
    for idx, (patch, noise, gaussian, diffusion) in enumerate(dataloader):
        patch = patch.numpy()
        noise = noise.numpy()
        gaussian = gaussian.numpy()
        diffusion = diffusion.numpy()
        print(idx, patch.dtype, noise.dtype, gaussian.dtype, diffusion.dtype)

        patch = patch[0][0]
        noise = noise[0][0]
        gaussian = gaussian[0][0]
        diffusion = diffusion[0][0]

        img = np.hstack([patch, noise, gaussian, diffusion])
        img = img * opt.minmax
        img = (img.copy() + 30.) * (255./60.)
        img = np.clip(img, 0, 255).astype(np.uint8)

        plot = plt.imshow(img, cmap='gray', animated=True)
        imgs.append([plot])

        if idx == 100 :
            break

    animate = animation.ArtistAnimation(fig, imgs, interval=500, blit=True)
    animate.save('img.gif')
    plt.close()