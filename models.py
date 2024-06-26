from nets.Unet import UNet
from nets.Unet_melfunc import UNet as melfunc
from nets.smaat_unet import SMAT_unet
from nets.TransUnet import TransUnet
from nets.vit_patch28 import VIT
from nets.Resnet import Resnet
from nets.ResUnet import ResUnet
from nets.ResUnetPlus import ResUnetPlusPlus
from nets.Unet import UNetSmall
from log import log_warning


def get_model(cfg):
    model_name = cfg.fit.model
    img_size = cfg.dataset.img_size

    """
    :param model_name: model name
    :param img_size: image size
    :return: model
    """

    if model_name == "unet":
        model = UNet()
    elif model_name == "smat":
        model = SMAT_unet()
    elif model_name == "transunet":
        model = TransUnet()
    elif model_name == "vit":
        model = VIT()
    elif model_name == "resnet":
        model = Resnet(img_size)
    elif model_name == "resunet":
        model = ResUnet()
    elif model_name == "resunetplus":
        model = ResUnetPlusPlus(1)
    elif model_name == "unetmelfunc":
        model = melfunc(drop_out=0.2)
    elif model_name == "smallunet":
        model = UNetSmall()
    else:
        log_warning("pls implemented model")
        raise NotImplementedError("implement a custom model(%s)" % model_name)

    return model.cuda()
