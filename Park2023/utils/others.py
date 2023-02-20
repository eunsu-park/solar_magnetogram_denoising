import torch.nn as nn


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