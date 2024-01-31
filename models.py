from models.Unet import UNet
from models.smaat_unet import SMAT_unet
from models.TransUnet import TransUnet
from models.vit_patch28 import VIT
from models.Resnet import Resnet


def get_model(cfg):
    model_name = cfg.fit.model
    img_size = cfg.dataset.img_size

    """
    :param model_name: model name
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

    return model.cuda()
