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
import os, time
from imageio import imsave

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
cuda = torch.cuda.device_count() > 0
ngpu = torch.cuda.device_count()
device = torch.device('cuda' if cuda else 'cpu')
print(cuda, ngpu)


path_model = os.path.join(opt.root_save, opt.name_data, "model")
path_snap = os.path.join(opt.root_save, opt.name_data, "snap")
os.makedirs(path_model, exist_ok=True)
os.makedirs(path_snap, exist_ok=True)

path_logger = "./%s.log" % (opt.name_data)
logger = open(path_logger, "w")
logger.close()


## Load Dataset ## 
from utils.others import get_loss_function, define_dataset_and_model
dataloader, network = define_dataset_and_model(opt)
if ngpu > 1 :
    network = nn.DataParallel(network)
network.to(device)

def lambda_rule(epoch):
    return 1.0 - max(0, epoch + 1 - opt.nb_epochs) / float(opt.nb_epochs_decay + 1)    

optim = torch.optim.Adam(network.parameters(),
    lr=opt.lr, betas=(opt.beta1, opt.beta2),
    eps=opt.eps, weight_decay=opt.weight_decay)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optim, lr_lambda=lambda_rule)

loss_function = get_loss_function(opt.loss_type).to(device)
metric_function = get_loss_function(opt.metric_type).to(device)

network.train()

@torch.no_grad()
def generation(model, inp):
    return model(inp)

palette = "Epoch: %d/%d Iteration: %d Loss: %5.3f Metric: %5.3f  Time: %dsec/%diters \n"
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
        noise = network(inp)
        gen = inp - noise
        loss = loss_function(gen, tar)
        metric = metric_function(gen, tar)
        loss.backward()
        optim.step()
        losses.append(loss.item())
        metrics.append(metric.item())

        iters += 1

        if iters % opt.report_freq == 0 :
            logger = open(path_logger, 'a')
            paint = (epochs, epochs_max, iters,
                np.mean(losses), np.mean(metrics), 
                time.time()-t0, opt.report_freq)
            print(palette % paint)
            logger.write(palette % paint)
            logger.close()

            network.eval()

            inp = patch_.clone().to(device)

            noise = generation(network, inp)
            inp = inp.cpu().numpy()[0][0]
            noise = noise.cpu().numpy()[0][0]
            gen = inp - noise

            snap = np.hstack([inp, gen, noise])
            snap = snap * opt.minmax
            snap = (snap + 30.) * (255./60.)
            snap = np.clip(snap, 0, 255).astype(np.uint8)
            imsave("./%s_latest.png" % (opt.name_data), snap)
            
            network.train()

            losses = []
            metrics = []
            t0 = time.time()

    epochs += 1
    scheduler.step()


    network.eval()

    inp = patch_.clone().to(device)

    noise = generation(network, inp)
    inp = inp.cpu().numpy()[0][0]
    noise = noise.cpu().numpy()[0][0]
    gen = inp - noise

    snap = np.hstack([inp, gen, noise])
    snap = snap * opt.minmax
    snap = (snap + 30.) * (255./60.)
    snap = np.clip(snap, 0, 255).astype(np.uint8)
    imsave("%s/%04d.png" % (path_snap, epochs), snap)
    
    network.train()


    if epochs % opt.save_freq == 0 :

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

