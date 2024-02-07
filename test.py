import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import wandb


# 1. 가장 로스 작은 사진 , 가장 적은 사진 보여주기
def visible_test(model, data_loaders):
    with torch.no_grad():
        for i in data_loaders[1]:
            show(i[1])
            show(model(i[0]))


def cal_loss_test(model, criterion, cfg, data_loaders):
    test_loss = test(model, criterion, cfg, data_loaders)
    print(f'test_score is {test_loss}')


def test(model, criterion, dataloaders, wandb_flag: bool = True):
    step = "Val"
    tepoch = tqdm(dataloaders, decs=step, total=len(dataloaders))
    running_loss = 0.0
    with torch.no_grad():
        for inputs, target in tepoch:
            outputs = model(inputs)
            loss = criterion(outputs, target)

            if ~torch.isfinite(loss):
                continue
            running_loss += loss.item()

            tepoch.set_postfix({'': 'loss : %.4f |' % (running_loss / tepoch.__len__())})

        if wandb_flag:
            wandb.log({step + "_loss": running_loss / tepoch.__len__()})

        return running_loss / tepoch.__len__()


def show(tensor):
    assert len(tensor.shape) == 4
    tensor = np.squeeze(np.array(tensor.detach().cpu()), axis=1)
    for i in range(tensor.shape[0]):
        plt.imshow(tensor[i])
        plt.show()
