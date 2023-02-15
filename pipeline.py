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

    def get_noise(self, size):
        scale = np.random.uniform(
            self.scale_min, self.scale_max)
        noise = np.random.normal(loc=0, scale=scale, size=size)
        return noise

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


#        print(idx, inp.shape, inp.min(), inp.max(), tar.min(), tar.max(), noise.min(), noise.max())

# def get_noise(mean, std, size):
#     noise = np.random.normal(loc=mean, scale=std, size=size)
#     return noise

# def get_diffusion_index(beta_start, beta_end, timesteps):

#     betas = np.linspace(beta_start, beta_end, timesteps)
#     alphas = 1. - betas
#     alphas_cumprod = np.cumprod(alphas, axis=0)
#     alphas_cumprod_prev = np.pad(alphas_cumprod[:-1], (1, 0), mode="constant", constant_values=1.0)

#     sqrt_recip_alphas = np.sqrt(1.0 / alphas)

#     # calculations for diffusion q(x_t | x_{t-1}) and others
#     sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
#     sqrt_one_minus_alphas_cumprod = np.sqrt(1. - alphas_cumprod)

#     # calculations for posterior q(x_{t-1} | x_t, x_0)
#     posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
#     return sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod

# def extract(a, t, x_shape):
#     batch_size = t.shape[0]
#     out = np.take(a, t, -1)
#     return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

# def get_diffusion(data_start, t, noise=None):
#     if noise is None:
#         noise = get_noise(0, 1, data_start.shape)

#     sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = get_diffusion_index(0.0001, 0.02)

#     sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, data_start.shape)
#     sqrt_one_minus_alphas_cumprod_t = extract(
#         sqrt_one_minus_alphas_cumprod, t, data_start.shape
#     )

#     return sqrt_alphas_cumprod_t * data_start + sqrt_one_minus_alphas_cumprod_t * noise


# class BaseDataset(data.Dataset):
#     def __init__(self, opt):
#         self.list_data = glob('%s/fits/*.fits'%(opt.root_data))
#         self.nb_data = len(self.list_data)
#         self.minmax = opt.minmax
#         self.patch_size = opt.patch_size
#         self.nb_random_patch = opt.nb_random_patch

#     def __len__(self):
#         return self.nb_data

#     def read_fits(self, fits):
#         return Map(fits).data.astype(np.float64)

#     def random_crop(self, data):
#         x, y = np.random.randint(2048-512, 2048+512-self.patch_size, 2)
#         return data[x:x+self.patch_size, y:y+self.patch_size]

#     def normalize(self, data):
#         return data / self.minmax

#     def numpy2torch(self, data):
#         return torch.from_numpy(data.astype(np.float32))


# class SingleNormalDataset(BaseDataset):
#     def __init__(self, opt):
#         super(SingleNormalDataset, self).__init__(opt)
#         self.noise_min = opt.noise_min
#         self.noise_max = opt.noise_max

#     def get_noise(self, size):
#         scale = np.random.uniform(
#             self.noise_min, self.noise_max)
#         noise = get_noise(0, scale, size)
#         return noise

#     def add_noise(self, data):
#         tar = data
#         noise = self.get_noise(tar.shape)
#         inp = tar + noise
#         return inp, tar, noise

#     def __getitem__(self, idx):
#         data = self.read_fits(self.list_data[idx])
#         inps = list()
#         tars = list()
#         noises = list()
#         for _ in range(self.nb_random_patch):
#             patch = self.random_crop(data)[None, None, :, :]
#             inp, tar, noise = self.add_noise(patch)
#             inps.append(inp)
#             tars.append(tar)
#             noises.append(noise)

#         inps = np.concatenate(inps, 0)
#         tars = np.concatenate(tars, 0)
#         noises = np.concatenate(noises, 0)

#         inps = self.normalize(inps)
#         tars = self.normalize(tars)
#         noises = self.normalize(noises)

#         inps = self.numpy2torch(inps)
#         tars = self.numpy2torch(tars)
#         noises = self.numpy2torch(noises)

#         return inps, tars, noises


# class MultiNormalDataset(BaseDataset):
#     def __init__(self, opt):
#         super(MultiNormalDataset, self).__init__(opt)
#         self.noisesteps = opt.noisesteps
#         self.noise_min = opt.noise_min
#         self.noise_max = opt.noise_max

#     def get_noise(self, size):
#         scale = np.random.uniform(
#             self.noise_min, self.noise_max)
#         noise = get_noise(0, scale, size)
#         return noise
        
#     def add_multistep_noise(self, inp, t):
#         for _ in range(t + 1):
#             tar = inp
#             noise = self.get_noise(tar.shape)
#             inp = tar + noise
#         return inp, tar, noise

#     def __getitem__(self, idx):
#         data = self.read_fits(self.list_data[idx])
#         inps = list()
#         tars = list()
#         noises = list()

#         for _ in range(self.nb_random_patch):
#             patch = self.random_crop(data)[None, None, :, :]
#             t = np.random.randint(self.noisesteps)
#             inp, tar, noise = self.add_multistep_noise(patch, t)
#             inps.append(inp)
#             tars.append(tar)
#             noises.append(noise)

#         inps = np.concatenate(inps, 0)
#         tars = np.concatenate(tars, 0)
#         noises = np.concatenate(noises, 0)

#         inps = self.normalize(inps)
#         tars = self.normalize(tars)
#         noises = self.normalize(noises)

#         inps = self.numpy2torch(inps)
#         tars = self.numpy2torch(tars)
#         noises = self.numpy2torch(noises)

#         return inps, tars, noises


# class DiffusionDataset(BaseDataset):
#     def __init__(self, opt):
#         super(DiffusionDataset, self).__init__(opt)
#         self.diffusionsteps = opt.diffusionsteps
#         self.beta_start = opt.beta_start
#         self.beta_end = opt.beta_end
#         self.build_diffusion()

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


# class UnifiedDataset(BaseDataset):
#     def __init__(self, opt):
#         super(UnifiedDataset, self).__init__(opt)
#         self.noisesteps = opt.noisesteps
#         self.noise_min = opt.noise_min
#         self.noise_max = opt.noise_max
#         self.diffusionsteps = opt.diffusionsteps
#         self.beta_start = opt.beta_start
#         self.beta_end = opt.beta_end
#         self.build_diffusion()

#     def get_noise(self, size):
#         scale = np.random.uniform(
#             self.noise_min, self.noise_max)
#         noise = get_noise(0, scale, size)
#         return noise

#     def add_noise(self, data):
#         inp = data.copy()
#         tar = data.copy()
#         noise = self.get_noise(data.shape)
#         inp += noise
#         return inp, tar, noise

#     def add_multistep_noise(self, data, t):
#         inp = data.copy()
#         for _ in range(t + 1):
#             tar = inp.copy()
#             noise = self.get_noise(data.shape)
#             inp = tar.copy() + noise
#         return inp, tar, noise

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

#     def get_single_normal_data(self, patch):
#         inp, tar, noise = self.add_noise(patch)
#         inp = self.normalize(inp)
#         tar = self.normalize(tar)
#         noise = self.normalize(noise)
#         return inp, tar, noise

#     def get_multi_normal_data(self, patch):
#         t = np.random.randint(self.noisesteps)
#         inp, tar, noise = self.add_multistep_noise(patch, t)
#         inp = self.normalize(inp)
#         tar = self.normalize(tar)
#         noise = self.normalize(noise)
#         return inp, tar, noise

#     def get_diffusion_data(self, patch):
#         t = np.random.randint(self.diffusionsteps)
#         patch = self.normalize(patch)
#         tar = patch.copy()
#         noise = np.random.normal(0, 1, patch.shape)
#         inp = self.add_diffusion(patch.copy(), t, noise)
#         return inp, tar, noise

#     def __getitem__(self, idx):
#         data = self.read_fits(self.list_data[idx])

#         single_inps = list()
#         single_tars = list()
#         single_noises = list()

#         multi_inps = list()
#         multi_tars = list()
#         multi_noises = list()

#         diffusion_inps = list()
#         diffusion_tars = list()
#         diffusion_noises = list()

#         for _ in range(self.nb_random_patch):
#             patch = self.random_crop(data)[None, None, :, :]
#             single_inp, single_tar, single_noise = self.get_single_normal_data(patch)
#             single_inps.append(single_inp)
#             single_tars.append(single_tar)
#             single_noises.append(single_noise)

#             multi_inp, multi_tar, multi_noise = self.get_multi_normal_data(patch)
#             multi_inps.append(multi_inp)
#             multi_tars.append(multi_tar)
#             multi_noises.append(multi_noise)

#             diffusion_inp, diffusion_tar, diffusion_noise = self.get_diffusion_data(patch)
#             diffusion_inps.append(diffusion_inp)
#             diffusion_tars.append(diffusion_tar)
#             diffusion_noises.append(diffusion_noise)

#         single_inps = self.numpy2torch(np.concatenate(single_inps, 0))
#         single_tars = self.numpy2torch(np.concatenate(single_tars, 0))
#         single_noises = self.numpy2torch(np.concatenate(single_noises, 0))

#         multi_inps = self.numpy2torch(np.concatenate(multi_inps, 0))
#         multi_tars = self.numpy2torch(np.concatenate(multi_tars, 0))
#         multi_noises = self.numpy2torch(np.concatenate(multi_noises, 0))

#         diffusion_inps = self.numpy2torch(np.concatenate(diffusion_inps, 0))
#         diffusion_tars = self.numpy2torch(np.concatenate(diffusion_tars, 0))
#         diffusion_noises = self.numpy2torch(np.concatenate(diffusion_noises, 0))

#         return (single_inps, single_tars, single_noises), (multi_inps, multi_tars, multi_noises), (diffusion_inps, diffusion_tars, diffusion_noises)


# if __name__ == '__main__' :

#     from options.train_option import TrainOption
#     from imageio import imsave

#     opt = TrainOption().parse()

#     datasets = [SingleNormalDataset(opt),
#         MultiNormalDataset(opt), DiffusionDataset(opt)]

#     inps = []
#     tars = []
#     noises = []

#     for dataset in datasets :
#         dataloader = data.DataLoader(dataset, batch_size=1, num_workers=0)
#         print(len(dataset), len(dataloader))

#         for idx, (inp, tar, noise) in enumerate(dataloader):
#             print(idx, inp.shape, tar.shape, inp.dtype, tar.dtype)
#             if idx == 20 :
#                 break
#         inps.append(inp[0][0][0])
#         tars.append(tar[0][0][0])
#         noises.append(noise[0][0][0])

#     for n in range(3):
#         dif = noises[n] - (inps[n] - tars[n])
#         print(dif.sum())

#     inps = np.hstack(inps) * 1000.
#     tars = np.hstack(tars) * 1000.
#     noises = np.hstack(noises) * 1000.
#     checks = inps - tars
#     img = np.vstack([inps, tars, noises, checks])

#     img = (img + 30) * (255./60.)
#     img = np.clip(img, 0, 255).astype(np.uint8)
#     imsave("img.png", img)


#     dataset = UnifiedDataset(opt)
#     dataloader = data.DataLoader(dataset, batch_size=1, num_workers=0)
#     print(len(dataset), len(dataloader))

#     for idx, (single, multi, diffusion) in enumerate(dataloader):
#         single_inp, single_tar, single_noise = single
#         multi_inp, multi_tar, multi_noise = multi
#         diffusion_inp, diffusion_tar, diffusion_noise = diffusion
        
#         print(idx, single_inp.shape, single_tar.shape, single_noise.shape)
#         print(idx, multi_inp.shape, multi_tar.shape, multi_noise.shape)
#         print(idx, diffusion_inp.shape, diffusion_tar.shape, diffusion_noise.shape)

#         if idx == 20 :
#             break


# #(single_inps, single_tars, single_noises), (multi_inps, multi_tars, multi_noises), (diffusion_inps, diffusion_tars, diffusion_noises)

#     # inps = []
#     # tars = []

#     # tar = np.ones((256, 256))
#     # for _ in range(10):
#     #     noise = get_noise(0, 10, tar.shape)
#     #     inp = tar + noise
#     #     tars.append(tar)
#     #     inps.append(inp)
#     #     tar = inp

#     # for n in range(9):
#     #     inp = inps[n]
#     #     tar = tars[n+1]
#     #     dif = tar - inp
#     #     print(dif.sum())

#     # inps = np.hstack(inps)
#     # tars = np.hstack(tars)

#     # inps = (inps + 30) * (255./60.)
#     # tars = (tars + 30) * (255./60.)

#     # inps = np.clip(inps, 0, 255).astype(np.uint8)
#     # tars = np.clip(tars, 0, 255).astype(np.uint8)

#     # img = np.vstack([inps, tars])

#     # imsave("imgs.png", img)

