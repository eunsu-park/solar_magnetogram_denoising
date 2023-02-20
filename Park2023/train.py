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



path_model = os.path.join(opt.root_save, opt.prefix, "model")
path_snap = os.path.join(opt.root_save, opt.prefix, "snap")
os.makedirs(path_model, exist_ok=True)
os.makedirs(path_snap, exist_ok=True)


## Load Dataset ## 
from pipeline import GaussianDataset
dataset_train = GaussianDataset(opt, is_train=True)
dataloader_train = data.DataLoader(
    dataset_train, batch_size=opt.batch_size,
    num_workers=opt.num_workers, shuffle=True)
print(len(dataset_train), len(dataloader_train))

dataset_test = GaussianDataset(opt, is_train=False)
dataloader_test = data.DataLoader(
    dataset_test, batch_size=opt.batch_size,
    num_workers=opt.num_workers, shuffle=True)
print(len(dataset_test), len(dataloader_test))


## Load Model ## 
from models.pix2pix_unet import UnetGenerator as PUNet, init_weights

network = PUNet(opt.ch_inp, opt.ch_tar, 6, 64)
init_weights(network)

if ngpu > 1 :
    network = nn.DataParallel(network)

network.to(device)

@torch.no_grad()
def generation(model, inp):
    return model(inp)

def hstack(tarray, nb=2):
    narray = tarray.cpu().numpy()
    return np.hstack([narray[n][0] for n in range(nb)])


def lambda_rule(epoch):
    return 1.0 - max(0, epoch + 1 - opt.nb_epochs) / float(opt.nb_epochs_decay + 1)    

optim = torch.optim.Adam(network.parameters(),
    lr=opt.lr, betas=(opt.beta1, opt.beta2),
    eps=opt.eps, weight_decay=opt.weight_decay)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optim, lr_lambda=lambda_rule)

from utils.others import get_loss_function

loss_function = get_loss_function(opt.loss_type).to(device)
metric_function = get_loss_function(opt.metric_type).to(device)

network.train()

palette = "\nEpoch: %d/%d Iteration: %d Loss: %5.3f Metric: %5.3f  Time: %dsec/%diters"
iters = 0
epochs = 0
losses = []
metrics = []
t0 = time.time()

epochs_max = opt.nb_epochs + opt.nb_epochs_decay

while epochs < epochs_max :
    for idx, (patch_, noise_, gaussian_) in enumerate(dataloader):

        optim.zero_grad()
        inp = gaussian_.clone().to(device)
        tar = patch_.clone().to(device)
        gen = network(inp)
        loss = loss_function(gen, tar)
        metric = metric_function(gen, tar)
        loss.backward()
        optim.step()
        losses.append(loss.item())
        metrics.append(metric.item())

        iters += 1

        if iters % opt.report_freq == 0 :
            paint = (epochs, epochs_max, iters,
                np.mean(losses), np.mean(metrics), 
                time.time()-t0, opt.report_freq)
            print(palette % paint)

            inp = patch_.clone().to(device)
            gen = generation(network, inp)
            
            inp = hstack(inp)
            gen = hstack(gen)

            noise = inp - gen

            snap = np.vstack([inp, gen, noise])
            snap = snap * opt.minmax
            snap = (snap + 30.) * (255./60.)
            snap = np.clip(snap, 0, 255).astype(np.uint8)
            imsave("%s/%07d.png" % (path_snap, iters), snap)
            imsave("./%s_train_latest.png" % (opt.prefix), snap)

            losses = []
            metrics = []
            t0 = time.time()

    epochs += 1
    scheduler.step()

    if ngpu > 1 :
        state_network = network.module.state_dict()
    else :
        state_network = network.state_dict()
    state_optim = optim.state_dict()
    state_scheduler = scheduler.state_dict()

    state = {'network':state_network,
        'optimizer':state_optim,
        'scheduler':state_scheduler}

    torch.save(state, '%s/%04d.pt' % (path_model, epochs))

