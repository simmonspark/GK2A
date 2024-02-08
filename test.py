import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import wandb


# 1. 가장 로스 작은 사진 , 가장 적은 사진 보여주기
def visible_test(model, criterion, data_loader):
    best = float('inf')
    store_input = 0
    store_target = 0
    store_pred = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            pred = model(inputs)
            loss = criterion(inputs, targets)
            if (loss < best):
                best = loss
                store_input = inputs
                store_target = targets
                store_pred = pred
    show(store_input[0:4, :, :, :])
    show(store_target[0:4, :, :, :] * 100.0)
    show(store_pred[0:4, :, :, :] * 100.0)
    print(f'END || best_loss is {best}')


def cal_loss_test(model, criterion, data_loaders):
    test_loss = test(model, criterion, data_loaders)
    print(f'test_score is {test_loss}')


def test(model, criterion, dataloader):

    running_loss = []
    with torch.no_grad():
        for inputs, target in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, target)

            if ~torch.isfinite(loss):
                continue
            running_loss.append(loss.item())

        return sum(running_loss) / len(running_loss)


def show(tensor):
    assert len(tensor.shape) == 4
    tensor = np.squeeze(np.array(tensor.detach().cpu()), axis=1)
    for i in range(tensor.shape[0]):
        for_debug = tensor[i]
        plt.imshow(tensor[i])
        plt.show()


# 픽셀이 있는 부분은 True else False
# pred도 똑같이 적용
# 두 개 사진을 or 연산 이게 분모
# and 연산이 분자
def get_best_threshold(model, criterion, loader):
    best_iou = 0
    best_th = 0
    plot_store = []
    with torch.no_grad():
        for i in np.arange(0.0, 1.0, 0.1):
            ious = []
            for inputs, target in loader:
                pred = model(inputs)
                pred = torch.where((pred > i), 1.0, 0.0)
                target = torch.where((target > 0), 1.0, 0.0)
                combined_binary = torch.clamp(pred + target, max=1.0)
                bottom = combined_binary.sum()
                top_binary = pred * target
                top = top_binary.sum()
                iou = top / bottom
                ious.append(iou.detach().cpu())
            plot_store.append(sum(ious) / len(ious))
            if best_iou < (sum(ious) / len(ious)):
                best_iou = (sum(ious) / len(ious))
                best_th = i

        print(f'END! best_iou is {best_iou} & best_th is {best_th}')
        plt.title('ious')
        plot_store = np.array(plot_store)
        plt.plot(plot_store)
        plt.show()

        best = float('inf')
        store_input = 0
        store_target = 0
        store_pred = 0

        for inputs, targets in loader:
            pred = model(inputs)
            pred[pred < best_th] = 0
            loss = criterion(inputs, targets)
            if (loss < best):
                best = loss
                store_input = inputs
                store_target = targets
                store_pred = pred
        show(store_input[0:4, :, :, :])
        show(store_target[0:4, :, :, :] * 100.0)
        show(store_pred[0:4, :, :, :] * 100.0)
        print(f'END || best_loss is {best}')
