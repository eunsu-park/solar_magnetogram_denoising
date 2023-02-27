import torch.nn as nn


def get_loss_function(loss_type):
    if loss_type == "l1" :
        loss_function = nn.L1Loss()
    elif loss_type == "l2" :
        loss_function = nn.MSELoss()
    elif loss_type == "huber":
        loss_function = nn.HuberLoss()
    else :
        raise NotImplementedError()
    return loss_function