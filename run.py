import os
import gc
import torch
import wandb
from tqdm import tqdm
from utils import mkdir_p
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def run(model, optimizer, criterion, cfg, dataloaders):
    best_loss = 100000000
    if cfg.fit.model_save_flag:
        save_dir = cfg.fit.model_save_path + "/" + cfg.fit.model + "/"
        if not os.path.exists(save_dir):
            mkdir_p(save_dir)

    if cfg.fit.train_flag:
        try:
            for epoch in range(cfg.fit.epochs):
                train_fn(epoch, model, optimizer, criterion, dataloaders[0], cfg.wandb.flag)
                val_loss = val_fn(epoch, model, criterion, dataloaders[1], cfg.wandb.flag)

                if best_loss > val_loss:
                    best_loss = val_loss
                    if cfg.fit.model_save_flag:
                        torch.save(model.state_dict(), save_dir + "BEST_" + cfg.fit.model + "_" +
                                   cfg.dataset.train.date_from + "_" + cfg.dataset.train.date_to + "_imgsize"
                                   + str(cfg.dataset.img_size) + ".pt")
                        print('best_model_saved!')

                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.memory_allocated('cuda')

        except KeyboardInterrupt:
            if cfg.fit.model_save_flag:
                print("KeyboardInterrupt! model_saved! ")
                torch.save(model.state_dict(), model.state_dict(), save_dir + "LAST_" + cfg.fit.model
                           + "_" + cfg.dataset.train.date_from + "_" + cfg.dataset.train.date_to
                           + "_imgsize" + str(cfg.dataset.img_size) + ".pt")


def train_fn(epoch, model, optimizer, criterion, dataloaders, wandb_flag: bool = True):
    step = "Train"

    with tqdm(dataloaders, desc=step, total=len(dataloaders)) as tepoch:
        model.train()
        running_loss = 0.0

        for inputs, target in tepoch:
            tepoch.set_description(step + "%d" % epoch)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)

            if ~torch.isfinite(loss):
                continue
            loss.requires_grad_(True)
            loss.backward(retain_graph=True)
            running_loss += loss.item()

            optimizer.step()

            tepoch.set_postfix({'': 'loss : %.4f | ' % (running_loss / tepoch.__len__())})

        if wandb_flag:
            wandb.log({"Train" + "_loss": running_loss / tepoch.__len__()}, step=epoch)


def val_fn(epoch, model, criterion, dataloaders, wandb_flag: bool = True):
    step = "Val"

    np_outputs = np.array([])
    np_target = np.array([])
    with tqdm(dataloaders, desc=step, total=len(dataloaders)) as tepoch:
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, target in tepoch:
                tepoch.set_description(step + "%d" % epoch)
                outputs = model(inputs)
                loss = criterion(outputs, target)

                if ~torch.isfinite(loss):
                    continue
                running_loss += loss.item()

                tepoch.set_postfix({'': 'loss : %.4f |' % (running_loss / tepoch.__len__())})

                np_outputs = np.append(np_outputs, outputs.detach().cpu().numpy().reshape(-1, ))
                np_target = np.append(np_target, target.detach().cpu().numpy().reshape(-1, ))

            np_outputs = np_outputs.reshape(-1, ) * 100.0
            np_target = np_target.reshape(-1, ) * 100.0
            # 이거 컨피그로 하고싶은데.. 데이터 겨울꺼 다 받고 컨피그 파일 수정할게염 푸쉬하면 귀찮으실까바 ㅎ

            POD, FAR, CSI = get_score(get_cf(np_outputs, np_target, 0.5))

            if wandb_flag:
                wandb.log({step + "_loss": running_loss / tepoch.__len__()}, step=epoch)
                wandb.log({"MAE_loss": MAE(np_outputs, np_target)}, step=epoch)
                wandb.log({"RMSE_loss": RMSE(np_outputs, np_target)}, step=epoch)
                wandb.log({"Corr_loss": corr(np_outputs, np_target)}, step=epoch)
                wandb.log({"POD": POD}, step=epoch)
                wandb.log({"FAR": FAR}, step=epoch)
                wandb.log({"CSI": CSI}, step=epoch)

        return running_loss / tepoch.__len__()


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


def MAE(pred, label):
    return np.mean(np.abs(pred - label))


def RMSE(pred, label):
    return np.sqrt(np.mean((pred - label) ** 2))


def corr(pred, label):
    return np.corrcoef(pred, label)[0, 1]


def get_cf(pred, label, th):
    pred = np.where(pred > th, 1, 0)
    label = np.where(label > th, 1, 0)
    cf = confusion_matrix(pred, label)
    return cf


def get_score(cf):
    TN = cf[0][0]
    FP = cf[0][1]
    FN = cf[1][0]
    TP = cf[1][1]

    POD = TP / (TP + FN)
    FAR = FP / (TP + FN)
    CSI = TP / (TP + FN + FP)

    return POD, FAR, CSI
