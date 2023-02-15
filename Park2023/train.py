from options.train_option import TrainOption

import numpy as np
import random
import torch

opt = TrainOption().parse()
np.random.seed(opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


import torch.nn as nn
from torch.utils import data as data

import os, time
from imageio import imsave

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

network_simple = PUNet(opt.ch_inp, opt.ch_tar, 6, 64)
network_diffusion = PUNet(opt.ch_inp, opt.ch_tar, 6, 64)

init_weights(network_simple)
init_weights(network_diffusion)

if ngpu > 1 :
    network_simple = nn.DataParallel(network_simple)
    network_diffusion = nn.DataParallel(network_diffusion)

network_simple.to(device)
network_diffusion.to(device)

@torch.no_grad()
def generation(model, inp):
    return model(inp)

def hstack(tarray, nb=4):
    narray = tarray.cpu().numpy()
    return [narray[n][0] for n in range(nb)] 


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
    for idx, (patch_, gaussian_, diffusion_) in enumerate(dataloader):

        optim_simple.zero_grad()
        inp = gaussian_.clone().to(device)
        tar = patch_.clone().to(device)
        gen = network_simple(inp)
        loss = loss_function(gen, tar)
        metric = metric_function(gen, tar)
        loss.backward()
        optim_simple.step()
        losses_simple.append(loss.item())
        metrics_simple.append(metric.item())

        optim_diffusion.zero_grad()
        inp = diffusion_.clone().to(device)
        tar = patch_.clone().to(device)
        gen = network_diffusion(inp)
        loss = loss_function(gen, tar)
        metric = metric_function(gen, tar)
        loss.backward()
        optim_simple.step()
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

            inp = patch_.clone().to(device)
            gen_simple = generation(network_simple, inp)
            gen_diffusion = generation(network_diffusion, inp)
            
            inp = hstack(inp)
            gen_simple = hstack(gen_simple)
            gen_diffusion = hstack(gen_diffusion)

            noise_simple = inp - gen_simple
            noise_diffusion = inp - gen_diffusion

            snap = np.vstack([inp, gen_simple, noise_simple, gen_diffusion, noise_diffusion])
            snap = snap * opt.minmax
            snap = (snap + 30.) * (255./60.)
            snap = np.clip(snap, 0, 255).astype(np.uint8)
            imsave("./train_latest.png", snap)

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

