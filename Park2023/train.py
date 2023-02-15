from options.train_option import TrainOption

import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils import data

import os, time
from imageio import imsave
import matplotlib.pyplot as plt

import math
from inspect import isfunction
from functools import partial

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F



opt = TrainOption().parse()

np.random.seed(opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
cuda = torch.cuda.device_count() > 0
ngpu = torch.cuda.device_count()
device = torch.device('cuda' if cuda else 'cpu')
print(cuda, ngpu)

path_model = "%s/model" % (opt.root_save)
path_snap = "%s/snap" % (opt.root_save)

os.makedirs(path_model, exist_ok=True)
os.makedirs(path_snap, exist_ok=True)


## Load Dataset ## 

from pipeline import BaseDataset

dataset = BaseDataset(opt)
dataloader = data.DataLoader(
    dataset, batch_size=opt.batch_size,
    num_workers=opt.num_workers, shuffle=True)
print(len(dataset), len(dataloader))

## Load Model ## 

from models.pix2pix_unet import UnetGenerator as PUNet, init_weights
from models.diffusion_unet import Unet as DUNet

network_simple = PUNet(opt.ch_inp, opt.ch_tar, 6, 64)
network_diffusion = DUNet(dim=opt.patch_size, channels=opt.ch_inp, dim_mults=(1, 2, 4,))

init_weights(network_simple)
init_weights(network_diffusion)

if ngpu > 1 :
    network_simple = nn.DataParallel(network_simple)
    network_diffusion = nn.DataParallel(network_diffusion)

network_simple.to(device)
network_diffusion.to(device)

## Diffusion Setting ## 

from utils.beta_schedules import linear_beta_schedule, cosine_beta_schedule, quadratic_beta_schedule, sigmoid_beta_schedule
betas = linear_beta_schedule(timesteps=opt.diffusionsteps)

from utils.diffusion import get_params, extract
sqrt_recip_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance = get_params(betas)


def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model, img):
    device = next(model.parameters()).device

    b = img.shape[0]
    # start from pure noise (for each example in the batch)
    #img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, opt.diffusionsteps)), desc='sampling loop time step', total=opt.diffusionsteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs

@torch.no_grad()
def sample(model, img):
    return p_sample_loop(model, img)


def lambda_rule(epoch):
    return 1.0 - max(0, epoch + 1 - opt.nb_epochs) / float(opt.nb_epochs_decay + 1)    

optim_simple = torch.optim.Adam(network_simple.parameters(),
    lr=opt.lr, betas=(opt.beta1, opt.beta2),
    eps=opt.eps, weight_decay=opt.weight_decay)
scheduler_simple = torch.optim.lr_scheduler.LambdaLR(optimizer=optim_simple, lr_lambda=lambda_rule)

optim_diffusion = torch.optim.Adam(network_diffusion.parameters(),
    lr=opt.lr, betas=(opt.beta1, opt.beta2),
    eps=opt.eps, weight_decay=opt.weight_decay)
scheduler_diffusion = torch.optim.lr_scheduler.LambdaLR(optimizer=optim_diffusion, lr_lambda=lambda_rule)

from utils.others import get_loss_function

loss_function = get_loss_function(opt.loss_type).to(device)
metric_function = get_loss_function(opt.metric_type).to(device)

network_simple.train()
network_diffusion.train()

palette = "\nEpoch:%d/%d Iteration:%d Time: %dsec/%diters"
palette_simple = "Simple Model, Loss: %5.3f Metric: %5.3f"
palette_diffusion = "Diffusion Model, Loss: %5.3f Metric: %5.3f"
iters = 0
epochs = 0
losses_simple = []
metrics_simple = []
losses_diffusion = []
metrics_diffusion = []
t0 = time.time()

epochs_max = opt.nb_epochs + opt.nb_epochs_decay

while epochs < epochs_max :
    for idx, (patch_, noise_) in enumerate(dataloader):

        optim_simple.zero_grad()
        inp = (patch_.clone() + noise_.clone()).to(device)
        tar = (noise_.clone()).to(device)
        pred = network_simple(inp)
        loss = loss_function(pred, tar)
        metric = metric_function(pred, tar)
        loss.backward()
        optim_simple.step()
        losses_simple.append(loss.item())
        metrics_simple.append(metric.item())

        optim_diffusion.zero_grad()
        t = torch.randint(0, opt.diffusionsteps, (opt.batch_size,), device=device).long()
        batch = patch_.clone().to(device)
        tar = noise_.clone().to(device)
        inp = q_sample(x_start=batch, t=t, noise=tar)
        pred = network_diffusion(inp, t)
        loss = loss_function(pred, tar)
        metric = metric_function(pred, tar)
        loss.backward()
        optim_diffusion.step()
        losses_diffusion.append(loss.item())
        metrics_diffusion.append(metric.item())

        iters += 1

        if iters % opt.report_freq == 0 :
            paint = (epochs, epochs_max, iters,
                time.time()-t0, opt.report_freq)
            print(palette % paint)
            paint_simple = (np.mean(losses_simple), np.mean(metrics_simple))
            print(palette_simple % paint_simple)
            paint_diffusion = (np.mean(losses_diffusion), np.mean(metrics_diffusion))
            print(palette_diffusion % paint_diffusion)

            network_simple.eval()
            network_diffusion.eval()

            inp = (patch_[0:1].clone() + noise_[0:1].clone()).to(device)
            pred = network_simple(inp)
            denoised_simple = inp - pred

            inp = (patch_[0:1].clone() + noise_[0:1].clone()).to(device)
            noise = noise_.clone().to(device)
            denoised_diffusion_all = np.concatenate(sample(network_diffusion, inp), 0)

            inp = inp[0][0].detach().cpu().numpy()
            denoised_simple = denoised_simple[0][0].detach().cpu().numpy()
            denoised_diffusion = denoised_diffusion_all[0][0]

            snap = np.hstack([inp, denoised_simple, denoised_diffusion])
            snap = snap * opt.minmax
            snap = (snap + 30.) * (255./60.)
            snap = np.clip(snap, 0, 255).astype(np.uint8)
            imsave("./train_latest.png", snap)

            for n in range(opt.diffusionsteps):
                snap = denoised_diffusion_all[n][0]
                snap = snap * opt.minmax
                snap = (snap + 30.) * (255./60.)
                snap = np.clip(snap, 0, 255).astype(np.uint8)
                imsave("./diffusion_latest/%04d.png"%(n), snap)

            network_simple.train()
            network_diffusion.train()

            losses_simple = []
            metrics_simple = []
            losses_diffusion = []
            metrics_diffusion = []
            t0 = time.time()

    epochs += 1
    scheduler_simple.step()
    scheduler_diffusion.step()

    if ngpu > 1 :
        state_simple = {'network':network_simple.module.state_dict(),
            'optimizer':optim_simple.state_dict(),
            'scheduler':scheduler_simple.state_dict()}
        state_diffusion = {'network':network_diffusion.module.state_dict(),
            'optimizer':optim_diffusion.state_dict(),
            'scheduler':scheduler_diffusion.state_dict()}
    else :
        state_simple = {'network':network_simple.state_dict(),
            'optimizer':optim_simple.state_dict(),
            'scheduler':scheduler_simple.state_dict()}
        state_diffusion = {'network':network_diffusion.state_dict(),
            'optimizer':optim_diffusion.state_dict(),
            'scheduler':scheduler_diffusion.state_dict()}
    torch.save(state_simple, '%s/simple.%04d.pt' % (path_model, epochs))
    torch.save(state_diffusion, '%s/diffusion.%04d.pt' % (path_model, epochs))

