import torchvision.models as models
import torch
import torch.nn as nn
from utils import get_nc_list
from torch.utils.data import DataLoader
from dataset import Dataset
from losses.resnet_loss import RES_LOSS
from losses.unet_loss import UNET_LOSS
from losses.vit_loss import vit_loss
from models.vit_patch28 import VIT
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from models.Unet import UNet
import os
from torch import cuda
import torch, gc
import wandb

########################################
# 여기서 모델만 바꾸고, MODE 수정 후 돌리세염 #
########################################
MODEL_NAME = 'resnet' # resnet unet 나머지는 추가 예정
DEVICE = 'cuda'
LR = 1e-5
MODEL_SAVE_PATH = os.path.join('/media/sien/DATA/weight/',MODEL_NAME+'.pt')
EPOCH = 200
MODE = 'train' # train, test, hell(hard train)
LOAD = False
RESOLUTION = 224
BATCH_SIZE = 4 #using vit : using 16 # resnet : batch 4 # unet : batch 8  --> vram 6~7 정도 사용합니다.

wandb.init(project="날씨!", name="experiment_name", config={
    "learning_rate": LR,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCH,
})



if(MODEL_NAME) == 'resnet' :
    model = models.resnet152(pretrained=True)
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    model.fc = nn.Sequential(
        nn.Linear(in_features=2048, out_features=2048),
        nn.Linear(in_features=2048, out_features=4096),
        nn.Linear(in_features=4096, out_features=RESOLUTION * RESOLUTION)
    )
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    loss_fn = RES_LOSS()
    model = model.to(DEVICE)

if(MODEL_NAME) == 'unet' :
    model = UNet()
    model = model.to(DEVICE)
    loss_fn = UNET_LOSS()

if(MODEL_NAME) == 'vit' :
    model = VIT()
    model = model.to(DEVICE)
    loss_fn = vit_loss()

optim = torch.optim.Adam(model.parameters(),LR)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optim,
                                            lr_lambda=lambda epoch: 0.95 ** epoch,
                                            last_epoch=-1,
                                            verbose=True)


train_ir, train_rr, sample_ir, sample_rr, test_ir, test_rr = get_nc_list('/media/sien/DATA/DATA/dataset/GK2A')
train_ds = Dataset(train_ir,train_rr)
sample_ds = Dataset(sample_ir,sample_rr)
test_ds = Dataset(test_ir,test_rr)

train_loader = DataLoader(train_ds,batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds,batch_size=BATCH_SIZE)
sample_loader = DataLoader(sample_ds,batch_size=BATCH_SIZE)


if LOAD is True :
    check_point = torch.load(MODEL_SAVE_PATH)
    model.load_state_dict(check_point)
def train_step(dataloader):

    loop = tqdm(dataloader,leave=True)
    model.train()
    loss_list = []
    for x, y in loop:
        x=x.to(DEVICE)
        y=y.to(DEVICE)
        pred = model(x)
        loss = loss_fn(pred,y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        loop.set_postfix(loss=loss.item())
        loss_list.append(loss)

    #scheduler.step()
    wandb.log({"one epoch average loss": sum(loss_list)/len(loss_list)})
    return sum(loss_list)/len(loss_list)
def validation(dataloader):
    loop = tqdm(dataloader, leave=True)
    model.eval()
    loss_list = []
    for x, y in loop:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optim.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loop.set_postfix(loss=loss.item())
        loss_list.append(loss)
    return sum(loss_list) / len(loss_list)
if __name__ == '__main__':
    try:
        if MODE == 'train':
            for i in range(EPOCH):
                train_step(dataloader=train_loader)
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f'EPOCH : {EPOCH} CURRENT : {i + 1}')
                """if((i+1)%10 == 0):
                    val_loss = validation(test_loader)
                    print(f'validation_loss is : {val_loss}')"""
                gc.collect()
                torch.cuda.empty_cache()
                cuda.memory_allocated('cuda')
                cuda.max_memory_allocated('cuda')
                cuda.memory_reserved('cuda')
                cuda.max_memory_reserved('cuda')

        if MODE == 'hell':
            cnt = 0
            flag = 10000
            ep_cnt = 0
            while True:
                ep_cnt+=1
                loss = train_step(dataloader=train_loader)
                if (loss < 100):
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                    break
                if (loss < 10000 and loss < flag):
                    flag = loss
                if (loss < 10000 and loss > flag):
                    cnt += 1
                if (cnt > 5):
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                    break

                """if ((ep_cnt + 1) % 10 == 0):
                    val_loss = validation(test_loader)
                    print(f'validation_loss is : {val_loss}')"""
    except KeyboardInterrupt:
        print("KeyboardInterrupt! model_saved! ")
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    if MODE == 'test' :

        for x,y in test_loader:
            x=x.to(DEVICE)
            model = model.eval()
            pred = model(x)
            x_single = x[0].cpu().detach().numpy().squeeze()
            pred_single = pred[0].cpu().detach().numpy().squeeze()

            x_single = np.abs(np.reshape(x_single, (224, 224))+0.1)
            pred_single = np.abs(np.reshape(pred_single, (224, 224))-0.1)
            rgb_image = np.stack([x_single, pred_single, pred_single], axis=-1)
            plt.figure(figsize=(6, 6))
            plt.imshow(rgb_image)
            plt.title('Rain fall prediction')
            plt.show()


