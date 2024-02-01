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
        pred = torch.reshape(pred, (-1, 1, 224, 224))
        target = torch.reshape(target, (-1, 1, 224, 224))

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
