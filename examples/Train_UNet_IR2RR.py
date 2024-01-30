import os
import random

import gc
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb
from torch import cuda
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import get_config
from dataset import IR2RR_Dataset
from loss_fn import loss_fn as reg_loss
from models.Unet import UNet

SEED = 0

# for Reproducible model
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

generator = torch.Generator()
generator.manual_seed(SEED)

gc.collect()
torch.cuda.empty_cache()
cuda.memory_allocated('cuda')
cuda.max_memory_allocated('cuda')
cuda.memory_reserved('cuda')
cuda.max_memory_reserved('cuda')




if LOSS_MODE == 'reg':
    loss_fn = reg_loss()

if MODEL_NAME == 'unet':
    model = UNet()
    model = model.to(DEVICE)

optim = torch.optim.Adam(model.parameters(), LR)




def train_step(dataloader):
    loop = tqdm(dataloader, leave=True)
    model.train()
    loss_list = []
    for x, y in loop:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        pred = model(x)
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
    for x, y in loop:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        pred = model(x)
        loss = loss_fn(pred, y)
        loop.set_postfix(loss=loss.item())
        loss_list.append(loss)
    wandb.log({"validation_one_epoch_loss": sum(loss_list) / len(loss_list)})
    return sum(loss_list) / len(loss_list)


if __name__ == '__main__':
    cfg = get_config("./configs/ir2rr_config.yaml")

    train_dataset = IR2RR_Dataset(cfg.dataset.root_path,
                                  cfg.dataset.train.date_from,
                                  cfg.dataset.train.date_to,
                                  cfg.dataset.interval_minutes,
                                  cfg.dataset.img_size)

    eval_dataset = IR2RR_Dataset(cfg.dataset.root_path,
                                 cfg.dataset.eval.date_from,
                                 cfg.dataset.eval.date_to,
                                 cfg.dataset.interval_minutes,
                                 cfg.dataset.img_size)

    train_loader = DataLoader(train_dataset, batch_size=cfg.dataset.train.batch_size, shuffle=cfg.dataset.train.shuffle)
    eval_loader = DataLoader(eval_dataset, batch_size=cfg.dataset.eval.batch_size, shuffle=cfg.dataset.eval.shuffle)

    try:
        if MODE == 'train':
            schedule_loss = []
            patient = 0
            for i in range(EPOCH):
                train_step(dataloader=sample_loader)
                print(f'EPOCH : {EPOCH} CURRENT : {i + 1}')
                if (i + 1) % 5 == 0:
                    with torch.no_grad():
                        val_loss = validation(sample_loader)
                        schedule_loss.append(val_loss)
                        print(f'validation_loss is : {val_loss}')
                if len(schedule_loss) == 2:
                    preb_val_loss = schedule_loss.pop(0)
                    current_val_loss = schedule_loss[0]
                    if current_val_loss > preb_val_loss:
                        patient += 1
                        print(
                            f"[patient = {patient}]: preb_val = {preb_val_loss} current_val = {current_val_loss},best model 저장 완료! ")
                        torch.save(model.state_dict(), MODEL_SAVE_PATH)
                if patient == 2:
                    print(f'patient -> {patient} break train')
                    break

                gc.collect()
                torch.cuda.empty_cache()
                cuda.memory_allocated('cuda')

    except KeyboardInterrupt:
        print("KeyboardInterrupt! model_saved! ")
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
