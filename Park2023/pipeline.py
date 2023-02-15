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
        self.scale_min = opt.noise_min
        self.scale_max = opt.noise_max
        self.patch_size = opt.patch_size

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

    def get_gaussian_noise(self, size):
        scale = np.random.uniform(
            self.scale_min, self.scale_max)
        noise = np.random.normal(loc=0, scale=scale, size=size)
        return noise

    def build_diffusion(self):
        betas = np.linspace(self.beta_start, self.beta_end, self.diffusionsteps)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        self.sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1. - alphas_cumprod)

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = np.take(a, t, -1)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def add_diffusion(self, data_start, t, noise=None):
        if noise is None:
            noise = get_noise(0, 1, data_start.shape)
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, np.array([t]), data_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, np.array([t]), data_start.shape)
        return sqrt_alphas_cumprod_t * data_start + sqrt_one_minus_alphas_cumprod_t * noise




    def __getitem__(self, idx):
        data = self.read_fits(self.list_data[idx])
        patch = self.random_crop(data)[None, :, :]
        patch = self.normalize(patch)
        patch = self.numpy2torch(patch)

        noise = self.get_noise(patch.shape)
        noise = self.normalize(noise)
        noise = self.numpy2torch(noise)
        return patch, noise

#     def build_diffusion(self):
#         betas = np.linspace(self.beta_start, self.beta_end, self.diffusionsteps)
#         alphas = 1. - betas
#         alphas_cumprod = np.cumprod(alphas, axis=0)
#         self.sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
#         self.sqrt_one_minus_alphas_cumprod = np.sqrt(1. - alphas_cumprod)

#     def extract(self, a, t, x_shape):
#         batch_size = t.shape[0]
#         out = np.take(a, t, -1)
#         return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

#     def add_diffusion(self, data_start, t, noise=None):
#         if noise is None:
#             noise = get_noise(0, 1, data_start.shape)
#         sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, np.array([t]), data_start.shape)
#         sqrt_one_minus_alphas_cumprod_t = self.extract(
#             self.sqrt_one_minus_alphas_cumprod, np.array([t]), data_start.shape)
#         return sqrt_alphas_cumprod_t * data_start + sqrt_one_minus_alphas_cumprod_t * noise

#     def __getitem__(self, idx):
#         data = self.read_fits(self.list_data[idx])
#         inps = list()
#         tars = list()
#         noises = list()

#         data = self.normalize(data)

#         for _ in range(self.nb_random_patch):
#             patch = self.random_crop(data)[None, None, :, :]
#             t = np.random.randint(self.diffusionsteps)
#             noise = np.random.normal(0, 1, patch.shape)
#             tar = patch
#             inp = self.add_diffusion(patch, t, noise)
#             inps.append(inp)
#             tars.append(tar)
#             noises.append(noise)

#         inps = np.concatenate(inps, 0)
#         tars = np.concatenate(tars, 0)
#         noises = np.concatenate(noises, 0)

#         inps = self.numpy2torch(inps)
#         tars = self.numpy2torch(tars)
#         noises = self.numpy2torch(noises)

#         return inps, tars, noises





if __name__ == "__main__" :

    from options.train_option import TrainOption

    opt = TrainOption().parse()

    dataset = BaseDataset(opt)
    dataloader = data.DataLoader(dataset, batch_size=4, num_workers=16)
    print(len(dataset), len(dataloader))

    for idx, (patch, noise) in enumerate(dataloader):
        patch = patch.numpy()
        noise = noise.numpy()
        print(idx, patch.dtype, noise.dtype)

        if idx == 20 :
            break

    from imageio import imsave

    patch = patch[0][0]
    noise = noise[0][0]
    inp = patch.copy() + noise
    tar = patch.copy()

    img = np.hstack([inp, tar, noise])
    print(img.shape)
    img = img * opt.minmax
    img = (img + 30.) * (255./60.)
    img = np.clip(img, 0, 255).astype(np.uint8)
    imsave("img.png", img)
