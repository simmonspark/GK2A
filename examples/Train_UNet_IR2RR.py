import wandb
import random
import datetime
import numpy as np

import gc
import torch
from torch import cuda
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from config import get_config
from dataset import IR2RR_Dataset
from models import get_model
from optim import optimizer
from loss_fn import loss_fn
from run import run

SEED = 0

# for Reproducible model
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
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

if __name__ == '__main__':
    cfg = get_config("../configs/ir2rr_config.yaml")

    train_dataset = IR2RR_Dataset(root_data_path=cfg.dataset.root_path,
                                  date_from=cfg.dataset.train.date_from,
                                  date_to=cfg.dataset.train.date_to,
                                  interval=cfg.dataset.train.interval_minutes,
                                  img_size=cfg.dataset.img_size)

    eval_dataset = IR2RR_Dataset(root_data_path=cfg.dataset.root_path,
                                 date_from=cfg.dataset.eval.date_from,
                                 date_to=cfg.dataset.eval.date_to,
                                 interval=cfg.dataset.eval.interval_minutes,
                                 img_size=cfg.dataset.img_size)

    train_loader = DataLoader(train_dataset, batch_size=cfg.dataset.train.batch_size, shuffle=cfg.dataset.train.shuffle)
    eval_loader = DataLoader(eval_dataset, batch_size=cfg.dataset.eval.batch_size, shuffle=cfg.dataset.eval.shuffle)
    data_loaders = [train_loader, eval_loader]

    model = get_model(cfg)
    optim = optimizer(model_params=model.parameters(), learning_rate=float(cfg.fit.learning_rate),
                      weight_decay=float(cfg.fit.weight_decay), optim=cfg.fit.optimizer)
    criterion = loss_fn(loss_name=cfg.fit.loss)

    if cfg.wandb.flag and cfg.fit.train_flag:
        wandb.init(project=cfg.wandb.project_name,
                   entity=cfg.wandb.entity,
                   name=cfg.fit.model + "/" +
                        cfg.dataset.train.date_from + "~" +
                        cfg.dataset.train.date_to + "/" +
                        cfg.dataset.eval.date_from + "~" +
                        cfg.dataset.eval.date_to + "/" +
                        str(cfg.dataset.img_size) + "/" +
                        datetime.datetime.now().strftime('%m-%d%H:%M:%S'))

        wandb.config = {"learning_rate": cfg.fit.learning_rate,
                        "epochs": cfg.fit.epochs,
                        "train_batch_size": cfg.dataset.train.batch_size,
                        "eval_batch_size": cfg.dataset.eval.batch_size}

        wandb.watch(model, log="all", log_freq=10)

    run(model, optim, criterion, cfg, data_loaders)

