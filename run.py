import os
import gc
import torch
import wandb
from tqdm import tqdm
from utils import mkdir_p

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

                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.memory_allocated('cuda')

        except KeyboardInterrupt:
            if cfg.fit.model_save_flag:
                print("KeyboardInterrupt! model_saved! ")
                torch.save(model.state_dict(), model.state_dict(), save_dir + "LAST_" + cfg.fit.model
                           + "_" + cfg.dataset.train.date_from + "_" + cfg.dataset.train.date_to
                           + "_imgsize" + str(cfg.fit.img_size) + ".pt")


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

            if wandb_flag:
                wandb.log({step + "_loss": running_loss / tepoch.__len__()}, step=epoch)

        return running_loss / tepoch.__len__()

