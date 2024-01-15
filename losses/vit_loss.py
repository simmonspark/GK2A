
from torch.nn import Module
import torch.nn as nn
import torch
torch.autograd.set_detect_anomaly(True)
class vit_loss(Module):
    def __init__(self):
        super(vit_loss,self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.expt_cor = 5.0
        self.background_cor = 1.0

    def forward(self,pred,target):

        #pred, target is 4d tensor : batched

        pred = torch.reshape(pred,(-1,224,224))
        target = torch.reshape(target,(-1,224,224))
        mask_expt = torch.sign(target)
        mask_background = 1-mask_expt

        expt_loss = self.mse(
            torch.flatten(pred*mask_expt,end_dim=-1),
            torch.flatten(target*mask_expt,end_dim=-1)
        )
        background_loss = self.mse(
            torch.flatten(pred*mask_background,end_dim=-1),
            torch.flatten(target * mask_background,end_dim=-1)
        )
        return (self.expt_cor * expt_loss) + (self.background_cor * background_loss)



