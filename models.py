from nets.Unet import UNet
from nets.smaat_unet import SMAT_unet
from nets.TransUnet import TransUnet
from nets.vit_patch28 import VIT
from nets.Resnet import Resnet
from log import log_warning


def get_model(cfg):
    model_name = cfg.fit.model
    img_size = cfg.dataset.img_size
    dropout_is = cfg.fit.dropout_is
    dropout_p = cfg.fit.dropout_p

    """
    :param model_name: model name
    :param img_size: image size
    :return: model
    """

    if model_name == "unet":
        if cfg.dataset.seq.is_seq is True:
            model = UNet(sequence=cfg.dataset.seq.sequence_len, drop_out=dropout_p, is_drop=dropout_is)
        else:
            model = UNet(drop_out=dropout_p, is_drop=dropout_is)
    elif model_name == "smat":
        model = SMAT_unet()
    elif model_name == "transunet":
        model = TransUnet()
    elif model_name == "vit":
        model = VIT()
    elif model_name == "resnet":
        model = Resnet(img_size)
    else:
        log_warning("pls implemented model")
        raise NotImplementedError("implement a custom model(%s)" % model_name)

    return model.cuda()
