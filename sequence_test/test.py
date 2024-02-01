
import os

import gc

import torch

import wandb
from torch import cuda
from torch.utils.data import DataLoader
from tqdm import tqdm

from sequence_dataset import Sampler_Dataset,SEQ_Sampler

from loss_fn import loss_fn as reg_loss
from model import UNet

save_root_path = '/media/sien/DATA/DATA/dataset'
date_from = '20230626'  # yyyyMMdd
date_to = '20230720'  # yyyyMMdd
interval_minutes = 10  # minutes

MODEL_NAME = 'unet'  # resnet unet vit transunet 나머지는 추가 예정
DEVICE = 'cuda'
LR = 5e-5
MODEL_SAVE_PATH = os.path.join('/media/sien/DATA/weight/', 'unet_seq' + '.pt')
EPOCH = 200
MODE = 'train'  # train, test, no_epoch(hard train)
LOAD = False
RESOLUTION = 224
LOSS_MODE = 'reg'
BATCH_SIZE = 32
SEQ_LEN = 3

torch.manual_seed(123)
torch.cuda.manual_seed(123)

gc.collect()
torch.cuda.empty_cache()
cuda.memory_allocated('cuda')
cuda.max_memory_allocated('cuda')
cuda.memory_reserved('cuda')
cuda.max_memory_reserved('cuda')

wandb.init(project="sequence_test", name='sequence_test', config={
    "learning_rate": LR,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCH,
})
if (LOSS_MODE == 'reg'):
    loss_fn = reg_loss()
if (MODEL_NAME) == 'unet':
    model = UNet(sequence=SEQ_LEN)
    model = model.to(DEVICE)
optim = torch.optim.Adam(model.parameters(), LR)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optim,
                                              lr_lambda=lambda epoch: 0.95 ** epoch,
                                              last_epoch=-1,
                                              verbose=True)

sample_ds = Sampler_Dataset('/media/sien/DATA/DATA/dataset','20230701','20230703',interval=10,img_size=256,sequence_len=SEQ_LEN,ir_only=False)
train_ds = Sampler_Dataset('/media/sien/DATA/DATA/dataset','20220626','20220710',interval=10,img_size=256,sequence_len=SEQ_LEN,ir_only=False)
sds_sampler = SEQ_Sampler(sample_ds.sampler_idx)
tds_sampler = SEQ_Sampler(train_ds.sampler_idx)
s_loader = DataLoader(dataset=sample_ds,sampler=sds_sampler,batch_size=BATCH_SIZE,shuffle=False)
t_loader = DataLoader(dataset=train_ds, sampler=tds_sampler, batch_size=BATCH_SIZE,shuffle=False)
if LOAD is True:
    check_point = torch.load(MODEL_SAVE_PATH)
    model.load_state_dict(check_point)
def train_step(dataloader):
    loop = tqdm(dataloader, leave=True)
    model.train()
    loss_list = []
    for img_seq in loop:

        img_seq = img_seq.to(DEVICE)
        y = img_seq[:, [SEQ_LEN-1], : , : ]
        pred = model(img_seq[:,:SEQ_LEN-1,:,:])



        loss = loss_fn(pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        loop.set_postfix(loss=loss.item())
        loss_list.append(loss)

    # scheduler.step()
    wandb.log({"one epoch average loss": sum(loss_list) / len(loss_list)})
    return sum(loss_list) / len(loss_list)


def validation(dataloader):
    loop = tqdm(dataloader, leave=True)
    model.eval()
    loss_list = []
    for img_seq in loop:
        img_seq = img_seq.to(DEVICE)
        pred = model(img_seq[:][SEQ_LEN - 1])
        y = img_seq[SEQ_LEN - 1]
        loss = loss_fn(pred, y)
        loop.set_postfix(loss=loss.item())
        loss_list.append(loss)
    wandb.log({"validation_one_epoch_loss": sum(loss_list) / len(loss_list)})
    return sum(loss_list) / len(loss_list)
if __name__ == '__main__':
    try:
        if MODE == 'train':
            schedule_loss = []
            patient = 0
            for i in range(EPOCH):
                train_step(dataloader=t_loader)
                print(f'EPOCH : {EPOCH} CURRENT : {i + 1}')
                if ((i + 1) % 5 == 0):
                    with torch.no_grad():
                        val_loss = validation(s_loader)
                        schedule_loss.append(val_loss)
                        print(f'validation_loss is : {val_loss}')
                if (len(schedule_loss) == 2):
                    preb_val_loss = schedule_loss.pop(0)
                    current_val_loss = schedule_loss[0]
                    if (current_val_loss > preb_val_loss):
                        patient += 1
                        print(
                            f"[patient = {patient}]: preb_val = {preb_val_loss} current_val = {current_val_loss},best model 저장 완료! ")
                        torch.save(model.state_dict(), MODEL_SAVE_PATH)
                if (patient == 2):
                    print(f'patient -> {patient} break train')
                    break

                gc.collect()
                torch.cuda.empty_cache()
                cuda.memory_allocated('cuda')


    except KeyboardInterrupt:
        print("KeyboardInterrupt! model_saved! ")
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    if MODE == 'test':

        for x, y in s_loader:
            x = x.to(DEVICE)
            model = model.eval()
            pred = model(x)
            x_single = x[0].cpu().detach().numpy().squeeze()
            pred_single = pred[0].cpu().detach().numpy().squeeze()


