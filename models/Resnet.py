import torch.nn as nn
import torchvision.models as models


def Resnet(img_size):
    model = models.resnet152(pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(in_features=2048, out_features=2048),
        nn.Linear(in_features=2048, out_features=4096),
        nn.Linear(in_features=4096, out_features=img_size * img_size)
    )
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return model
