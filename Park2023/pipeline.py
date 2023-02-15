import torch
from torch.utils import data
import numpy as np
from glob import glob
from sunpy.map import Map
import torch.nn.functional as F


class BaseDataset(data.Dataset):
    def __init__(self, opt):
        self.list_data = glob('%s/fits/*.fits'%(opt.root_data))
        self.nb_data = len(self.list_data)
        self.minmax = opt.minmax
        self.noise_loc = opt.noise_loc
        self.noise_scale = opt.noise_scale
        self.patch_size = opt.patch_size
        self.beta_start = opt.beta_start
        self.beta_end = opt.beta_end
        self.steps = opt.steps

        self.build_diffusion()

    def __len__(self):
        return self.nb_data

    def read_fits(self, fits):
        return Map(fits).data.astype(np.float64)

    def random_crop(self, data):
        x, y = np.random.randint(2048-512, 2048+512-self.patch_size, 2)
        return data[x:x+self.patch_size, y:y+self.patch_size]

    def normalize(self, data):
        return data / self.minmax

    def numpy2torch(self, data):
        return torch.from_numpy(data.astype(np.float32))

    def get_gaussian(self, size):
        return np.random.normal(loc=self.noise_loc, scale=self.noise_scale, size=size)

    def add_gaussian(self, data):
        noise = self.get_gaussian(data.shape)
        return noise + data

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

    def add_diffusion(self, data):
        t = np.random.randint(self.steps)
        noise = self.get_gaussian(data.shape)
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, np.array([t]), data.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, np.array([t]), data.shape)
        return sqrt_alphas_cumprod_t * data + sqrt_one_minus_alphas_cumprod_t * noise

    def __getitem__(self, idx):
        data = self.read_fits(self.list_data[idx])
        patch = self.random_crop(data)[None, :, :]
        gaussian = self.add_gaussian(patch.copy())
        diffusion = self.add_diffusion(patch.copy())

        patch = self.normalize(patch)
        patch = self.numpy2torch(patch)

        gaussian = self.normalize(gaussian)
        gaussian = self.numpy2torch(gaussian)

        diffusion = self.normalize(diffusion)
        diffusion = self.numpy2torch(diffusion)

        return patch, gaussian, diffusion

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

    fig = plt.figure()
    for idx, (patch, gaussian, diffusion) in enumerate(dataloader):
        patch = patch.numpy()
        gaussian = gaussian.numpy()
        diffusion = diffusion.numpy()
        print(idx, patch.dtype, gaussian.dtype, diffusion.dtype)

        patch = patch[0][0]
        gaussian = gaussian[0][0]
        diffusion = diffusion[0][0]
        img = np.hstack([patch, gaussian, diffusion])
        img = img * opt.minmax
        img = (img + 30.) * (255./60.)
        img = np.clip(img, 0, 255).astype(np.uint8)

        plot = plt.imshow(img, cmap='gray', animated=True)
        imgs.append([plot])

        if idx == 100 :
            break

    animate = animation.ArtistAnimation(fig, imgs, blit=True)
    animate.save('img.gif')
    plt.close()