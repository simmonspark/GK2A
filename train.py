import torchvision.models as models
import torch.nn as nn
from utils import get_nc_list
from torch.utils.data import DataLoader
from dataset import Dataset
from loss_fn import loss_fn as reg_loss
from models.vit_patch28 import VIT
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from models.Unet import UNet
import os
from torch import cuda
import torch, gc
import wandb
import random
import argparse
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from utils import create_classification_mask as ccm
from models.smaat_unet import SMAT_unet

########################################
# 여기서 모델만 바꾸고, MODE 수정 후 돌리세염 #
########################################
#


MODEL_NAME = 'smat' # resnet unet vit transunet 나머지는 추가 예정
DEVICE = 'cuda'
LR = 5e-5
MODEL_SAVE_PATH = os.path.join('/media/sien/DATA/weight/',MODEL_NAME+'.pt')
EPOCH = 200
MODE = 'train' # train, test, no_epoch(hard train)
LOAD = False
RESOLUTION = 224
LOSS_MODE = 'reg'# reg, class
#resnet batch4 -> 8g
#unet batch 8 -> 7g, batch 64 -> 22g
#vit : batch 4 -> 30g...
BATCH_SIZE = 32

torch.manual_seed(123)
torch.cuda.manual_seed(123)

gc.collect()
torch.cuda.empty_cache()
cuda.memory_allocated('cuda')
cuda.max_memory_allocated('cuda')
cuda.memory_reserved('cuda')
cuda.max_memory_reserved('cuda')

wandb.init(project="날씨!", name=MODEL_NAME, config={
    "learning_rate": LR,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCH,
})

if(LOSS_MODE == 'reg'):
    loss_fn = reg_loss()
if(LOSS_MODE == 'class'):
    print('이 로스는 망했습니다. ㅜㅜ')


if(MODEL_NAME) == 'resnet' :
    model = models.resnet152(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(in_features=2048, out_features=2048),
        nn.Linear(in_features=2048, out_features=4096),
        nn.Linear(in_features=4096, out_features=RESOLUTION * RESOLUTION)
    )
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model = model.to(DEVICE)

if(MODEL_NAME)== 'transunet' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='../data/Synapse/train_npz', help='root dir for data')
    parser.add_argument('--dataset', type=str,
                        default='Synapse', help='experiment_name')
    parser.add_argument('--list_dir', type=str,
                        default='./lists/lists_Synapse', help='list dir')
    parser.add_argument('--num_classes', type=int,
                        default=1, help='output channel of network')
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int,
                        default=150, help='maximum epoch number to train')
    '''
    여기 배치!
    '''
    parser.add_argument('--batch_size', type=int,
                        default=64, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')

    parser.add_argument('--base_lr', type=float, default=0.001,
                        help='segmentation network learning rate')
    parser.add_argument('--img_size', type=int,
                        default=224, help='input patch size of network input')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--n_skip', type=int,
                        default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_name', type=str,
                        default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int,
                        default=16, help='vit_patches_size, default is 16')
    args = parser.parse_args()
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': '../data/Synapse/train_npz',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 1,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(
        args.vit_patches_size) if args.vit_patches_size != 16 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
                                          0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
        int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

if(MODEL_NAME) == 'unet' :
    model = UNet()
    model = model.to(DEVICE)

if(MODEL_NAME) == 'smat' :
    model = SMAT_unet()
    model = model.to(DEVICE)

if(MODEL_NAME) == 'vit' :
    model = VIT()
    model = model.to(DEVICE)


optim = torch.optim.Adam(model.parameters(),LR)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optim,
                                            lr_lambda=lambda epoch: 0.95 ** epoch,
                                            last_epoch=-1,
                                            verbose=True)


train_ir, train_rr, sample_ir, sample_rr, test_ir, test_rr = get_nc_list('/media/sien/DATA/DATA/dataset/GK2A')
train_ds = Dataset(train_ir,train_rr)
sample_ds = Dataset(sample_ir,sample_rr)
test_ds = Dataset(test_ir,test_rr)

train_loader = DataLoader(train_ds,batch_size=BATCH_SIZE,num_workers=15)
test_loader = DataLoader(test_ds,batch_size=BATCH_SIZE,num_workers=15)
sample_loader = DataLoader(sample_ds,batch_size=BATCH_SIZE,num_workers=15)


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
        pred = model(x)
        loss = loss_fn(pred, y)
        loop.set_postfix(loss=loss.item())
        loss_list.append(loss)
    wandb.log({"validation_one_epoch_loss": sum(loss_list) / len(loss_list)})
    return sum(loss_list) / len(loss_list)
if __name__ == '__main__':
    try:
        if MODE == 'train':
            schedule_loss = []
            for i in range(EPOCH):
                train_step(dataloader=train_loader)
                print(f'EPOCH : {EPOCH} CURRENT : {i + 1}')
                if((i+1)%5 == 0):
                    with torch.no_grad():
                        val_loss = validation(sample_loader)
                        schedule_loss.append(val_loss)
                        print(f'validation_loss is : {val_loss}')
                if(len(schedule_loss)==2):
                    preb_val_loss = schedule_loss.pop(0)
                    current_val_loss = schedule_loss[0]
                    if(current_val_loss>preb_val_loss):
                        print(f"[stoped by scheduler]: preb_val = {preb_val_loss} current_val = {current_val_loss}, model saved and break ")
                        torch.save(model.state_dict(), MODEL_SAVE_PATH)
                        break

                gc.collect()
                torch.cuda.empty_cache()
                cuda.memory_allocated('cuda')

        if MODE == 'no_epoch':
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

                if ((ep_cnt + 1) % 5 == 0):
                    val_loss = validation(test_loader)
                    print(f'validation_loss is : {val_loss}')
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
            pred_single = np.abs(np.reshape(pred_single*100.0, (224, 224)))
            rgb_image = np.stack([x_single, pred_single, pred_single], axis=-1)
            plt.figure(figsize=(6, 6))
            plt.imshow(rgb_image)
            plt.title('Rain fall prediction')
            mask1,mask2,mask3,mask4 = ccm(pred_single)
            plt.show()


