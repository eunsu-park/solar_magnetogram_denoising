import torch.nn as nn
from torch.utils import data as data


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
    if opt.type_train == "autoencoder" :
        from pipeline import GaussianDataset as Dataset
        from models.pix2pix_unet import UnetGenerator as PUNet, init_weights
        dataset = Dataset(opt)
        network = PUNet(opt.ch_inp, opt.ch_tar, 6, 64)
        init_weights(network)
    dataloader = data.DataLoader(
        dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, shuffle=opt.is_train)
    print(len(dataset), len(dataloader))
    return dataloader, network