import torch.nn as nn
from torch.utils import data as data
import torch

def get_loss_function(loss_type, reduction='mean'):
    if loss_type == "l1" :
        loss_function = nn.L1Loss(reduction=reduction)
    elif loss_type == "l2" :
        loss_function = nn.MSELoss(reduction=reduction)
    elif loss_type == "huber":
        loss_function = nn.HuberLoss(reduction=reduction)
    else :
        raise NotImplementedError()
    return loss_function

def define_dataset_and_model(opt):
    from pipeline import GaussianDataset as Dataset
    from models.pix2pix_unet import UnetGenerator as PUNet, init_weights
    dataset = Dataset(opt)
    network = PUNet(opt.ch_inp, opt.ch_tar, opt.nb_down, 64, use_dropout=True)
    init_weights(network)
    dataloader = data.DataLoader(
        dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, shuffle=opt.is_train)
    print(len(dataset), len(dataloader))
    return dataloader, network

def define_optim_and_scheduler(opt, network):
    def lambda_rule(epoch):
        return 1.0 - max(0, epoch + 1 - opt.nb_epochs) / float(opt.nb_epochs_decay + 1)    
    optim = torch.optim.Adam(network.parameters(),
        lr=opt.lr, betas=(opt.beta1, opt.beta2),
        eps=opt.eps, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optim, lr_lambda=lambda_rule)
    return optim, scheduler