from torch.nn import Module
import torch.nn as nn
import torch

from log import log_warning

torch.autograd.set_detect_anomaly(True)


def loss_fn(loss_name):
    """
    :param loss_fn: implement loss function for training
    :return: loss function module(class)
    """

    if loss_name == "MSE":
        return sienMSE()
    if loss_name == "MAE":
        return sienMAE()
    if loss_name == "origin":
        return torch.nn.MSELoss()

    else:
        log_warning("use implemented loss functions")
        raise NotImplementedError("implement a custom function(%s) in loss.py" % loss_fn)


class sienMSE(Module):
    def __init__(self):
        super(sienMSE, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.expt_cor = 1.0
        self.background_cor = 0.5

    def forward(self, pred, target):
        # pred, target is 4d tensor : batched
        assert len(pred.shape)==4
        pred = torch.reshape(pred, (-1, 1, pred.shape[2], pred.shape[3]))
        target = torch.reshape(target, (-1, 1, target.shape[2], target.shape[3]))

        mask_expt = torch.sign(target)
        mask_background = 1 - mask_expt

        expt_loss = self.mse(
            torch.flatten(pred * mask_expt, end_dim=-1),
            torch.flatten(target * mask_expt, end_dim=-1)
        )
        background_loss = self.mse(
            torch.flatten(pred * mask_background, end_dim=-1),
            torch.flatten(target * mask_background, end_dim=-1)
        )
        return (self.expt_cor * expt_loss) + (self.background_cor * background_loss)


class sienMAE(Module):
    def __init__(self):
        super(sienMSE, self).__init__()
        self.mae = nn.L1Loss(reduction='sum')
        self.expt_cor = 1.0
        self.background_cor = 0.5

    def forward(self, pred, target):
        # pred, target is 4d tensor : batched
        pred = torch.reshape(pred, (-1, 1, 224, 224))
        target = torch.reshape(target, (-1, 1, 224, 224))

        mask_expt = torch.sign(target)
        mask_background = 1 - mask_expt

        expt_loss = self.mae(
            torch.flatten(pred * mask_expt, end_dim=-1),
            torch.flatten(target * mask_expt, end_dim=-1)
        )
        background_loss = self.mae(
            torch.flatten(pred * mask_background, end_dim=-1),
            torch.flatten(target * mask_background, end_dim=-1)
        )
        return (self.expt_cor * expt_loss) + (self.background_cor * background_loss)
