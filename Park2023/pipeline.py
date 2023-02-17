import torch
from torch.utils import data
import numpy as np
from glob import glob
from sunpy.map import Map
import torch.nn.functional as F
from torchvision.transform import Compose


class GenerateGaussian(object):
    def __init__(self, loc, scale, size):
        self.loc = loc
        self.scale = scale
        self.size = size
    def __call__(self):
        return np.random.normal(loc=self.loc, scale=self.scale, size=self.size)

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

class GaussianDataset(BaseDataset):
    def __init__(self, opt):
        super(GaussianDataset, self).__init__(opt)



class BaseDataset(data.Dataset):
    def __init__(self, opt):
        self.list_data = glob('%s/*.fits'%(opt.root_data))
        self.nb_data = len(self.list_data)
        self.steps = opt.steps
        self.patch_size = opt.patch_size
        self.beta_start = opt.beta_start
        self.beta_end = opt.beta_end

        self.random_crop = RandomCrop(opt.patch_size)
        self.generate_gaussian = GenerateGaussian(loc=opt.noise_loc, scale=opt.noise_scale, size=(opt.patch_size, opt.patch_size))
        self.compose = Compose([Normalize(opt.minmax), Cast()])

        self.build_diffusion()

    def __len__(self):
        return self.nb_data

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

    def get_diffusion(self, size):
        t = np.random.randint(self.steps)
        noise = self.generate_gaussian()
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, np.array([t]), size)
        return sqrt_one_minus_alphas_cumprod_t * noise

    def add_diffusion(self, data):
        t = np.random.randint(self.steps)
        noise = self.generate_gaussian()
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, np.array([t]), data.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, np.array([t]), data.shape)
        return sqrt_alphas_cumprod_t * data + sqrt_one_minus_alphas_cumprod_t * noise


    def __getitem__(self, idx):
        data = self.read_fits(self.list_data[idx])
        patch = self.random_crop(data)[None, :, :]

        gaussian_noise_single = self.generate_gaussian()
        gaussian_noise_stacked = np.sum([self.generate_gaussian() for _ in range(self.steps)], 0)
        diffusion_noise = self.get_diffusion(patch.shape)
        
        gaussian_single = patch.copy() + gaussian_noise_single.copy()
        gaussian_stacked = patch.copy() + gaussian_noise_stacked.copy()
        diffusion = patch.copy() + diffusion_noise.copy()

        patch = self.compose(patch)
        gaussian_simple = self.compose(gaussian_simple)
        gaussian_stacked = self.compose(gaussian_stacked)
        diffusion = self.compose(diffusion)

        return patch, gaussian_single, gaussian_stacked, diffusion







if __name__ == "__main__" :

    from options.train_option import TrainOption
    from imageio import imsave
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt


    opt = TrainOption().parse()

    dataset = BaseDataset(opt)
    dataloader = data.DataLoader(dataset, batch_size=4, num_workers=16)
    print(len(dataset), len(dataloader))

    imgs = []

    fig = plt.figure(figsize=(14, 3))
    for idx, (patch, gaussian, diffusion) in enumerate(dataloader):
        patch = patch.numpy()
        gaussian = gaussian.numpy()
        diffusion = diffusion.numpy()
        print(idx, patch.dtype, gaussian.dtype, diffusion.dtype)

        patch = patch[0][0]
        gaussian = gaussian[0][0]
        diffusion = diffusion[0][0]

        img = np.hstack([patch, gaussian, gaussian-patch, diffusion, diffusion-patch])
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